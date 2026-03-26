import csv
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read, write
from ase.optimize import FIRE

os.environ.setdefault("MATGL_BACKEND", "DGL")

import matgl
from matgl.ext._ase_dgl import PESCalculator

from ASE.oh_reaction_search_space import ADSORBATE_LIBRARY, allowed_neighbors
from ASE.screen_chgnet_intermediates import (
    A_TOL,
    AZIMUTHS,
    TILTS,
    add_case,
    apply_rotation,
    build_rotated_adsorbates,
    identify_surface_sites,
    place_adsorbate,
    rotation_matrix,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_CIF = ROOT / "intermediates" / "POSCAR.cif"
MODEL_DIR = ROOT / "matgl" / "pretrained_models" / "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"
OUT_DIR = ROOT / "intermediates" / "orr_search_chgnet_pbe"
SHORT_CONTACT_SCALE = 0.72
REASONABLE_FORCE_EV_A = 0.03
REASONABLE_WINDOW_EV = 0.6
EXP_STAGE1_STEPS = 8
EXP_STAGE1_FMAX = 0.35
EXP_FINAL_STEPS = 500
EXP_FINAL_FMAX = 0.01


def covalent_sum(symbol_a: str, symbol_b: str) -> float:
    return float(covalent_radii[atomic_numbers[symbol_a]] + covalent_radii[atomic_numbers[symbol_b]])


def has_bad_contacts(atoms: Atoms) -> tuple[bool, float]:
    positions = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    min_ratio = 999.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            ratio = dist / covalent_sum(syms[i], syms[j])
            min_ratio = min(min_ratio, ratio)
            if ratio < SHORT_CONTACT_SCALE:
                return True, min_ratio
    return False, min_ratio


def canonicalize_custom(species: str, mode: str) -> tuple[Atoms, int]:
    zhat = np.array([0.0, 0.0, 1.0])
    if species == "OOH":
        # Anchor O at origin, O-O points out of the surface, terminal H attached to distal O.
        oo = 1.46
        oh = 0.98
        angle = np.deg2rad(104.0)
        atoms = Atoms(
            "OOH",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, oo],
                [oh * np.sin(angle), 0.0, oo + oh * np.cos(angle)],
            ],
        )
        return atoms, 0
    if species == "H2O2":
        oo = 1.47
        oh = 0.98
        angle = np.deg2rad(100.0)
        atoms = Atoms(
            "H2O2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, oo],
                [oh * np.sin(angle), 0.0, oh * np.cos(angle)],
                [-oh * np.sin(angle), 0.0, oo - oh * np.cos(angle)],
            ],
        )
        return atoms, 0
    raise ValueError(f"Unsupported custom species: {species}")


def build_adsorbate_variants(species: str) -> list[tuple[str, Atoms, int]]:
    if species in {"O", "OH", "H2O"}:
        mode = "atom" if species == "O" else "co_bound"
        return build_rotated_adsorbates(species, mode, TILTS, AZIMUTHS)
    if species == "O2":
        variants = build_rotated_adsorbates("O2", "end_on", TILTS, AZIMUTHS)
        variants.extend(build_rotated_adsorbates("O2", "side_on", TILTS, AZIMUTHS))
        return variants
    if species in {"OOH", "H2O2"}:
        base, anchor_idx = canonicalize_custom(species, "default")
        variants = []
        for tilt in TILTS:
            curr_az = (0.0,) if abs(tilt) < 1e-12 else AZIMUTHS
            for azimuth in curr_az:
                ads = base.copy()
                if tilt:
                    apply_rotation(ads, rotation_matrix(np.array([1.0, 0.0, 0.0]), tilt), np.zeros(3))
                if azimuth:
                    apply_rotation(ads, rotation_matrix(np.array([0.0, 0.0, 1.0]), azimuth), np.zeros(3))
                variants.append((f"default_tilt{int(tilt)}_az{int(azimuth)}", ads, anchor_idx))
        return variants
    raise ValueError(f"Unsupported species: {species}")


def relax_candidate(atoms: Atoms, potential, steps: int, fmax: float) -> tuple[Atoms, dict]:
    atoms = deepcopy(atoms)
    atoms.calc = PESCalculator(potential=potential, stress_unit="eV/A3")
    initial_energy = float(atoms.get_potential_energy())
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    final_energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    return atoms, {
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "n_steps": int(getattr(opt, "nsteps", steps)),
        "max_force_eVA": float(np.linalg.norm(forces, axis=1).max()),
        "converged": bool(np.linalg.norm(forces, axis=1).max() <= fmax + 1e-12),
    }


def build_orr_candidates(base_slab: Atoms, info: dict) -> list[dict]:
    candidates = []
    slab_vac = base_slab.copy()
    del slab_vac[info["vacancy_idx"]]

    positions = base_slab.get_positions()
    species = np.array(base_slab.get_chemical_symbols())
    top_z = info["top_z"]
    top_mask = np.isclose(positions[:, 2], top_z, atol=A_TOL)
    top_o = [int(i) for i in np.where(top_mask & (species == "O"))[0]]

    primary_co = info["neighbor_co"][0]
    primary_co_xy = positions[primary_co, :2]
    intact_co_xy = positions[info["intact_co_site"], :2]

    def nearest_top_o_to(xy: np.ndarray, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        candidates_o = [i for i in top_o if i not in excluded]
        return min(candidates_o, key=lambda i: np.linalg.norm(positions[i, :2] - xy))

    direct_o_idx = nearest_top_o_to(intact_co_xy)
    direct_o_xy = positions[direct_o_idx, :2]
    direct_bridge_xy = 0.5 * (intact_co_xy + direct_o_xy)
    vacancy_edge_xy = 0.5 * (info["vacancy_xy"] + primary_co_xy)

    contexts = [
        ("direct_co_top", base_slab, intact_co_xy, top_z, {"site": "co_top"}),
        ("direct_hollow", base_slab, info["hollow_xy"], top_z, {"site": "hollow"}),
        ("direct_co_o_bridge", base_slab, direct_bridge_xy, top_z, {"site": "co_o_bridge", "co_idx": info["intact_co_site"], "o_idx": direct_o_idx}),
        ("vacancy_center", slab_vac, info["vacancy_xy"], info["vacancy_z"], {"site": "vacancy"}),
        ("vacancy_edge_bridge", slab_vac, vacancy_edge_xy, top_z, {"site": "vacancy_edge_bridge", "co_idx": primary_co}),
    ]

    for co_idx in info["neighbor_co"]:
        contexts.append(
            (
                f"vacancy_co_top_{co_idx}",
                slab_vac,
                positions[co_idx, :2],
                top_z,
                {"site": "neighbor_co_top", "co_idx": co_idx},
            )
        )

    height_by_species = {
        "O": 1.6,
        "OH": 1.8,
        "H2O": 2.0,
        "O2": 1.75,
        "OOH": 1.85,
        "H2O2": 1.95,
    }

    # Build a reaction graph so the output records search edges, even though we enumerate the current library.
    graph = {name: allowed_neighbors(name) for name in ADSORBATE_LIBRARY}

    for state_name in ("O", "OH", "H2O", "O2", "OOH", "H2O2"):
        for ctx_name, slab, xy, zref, metadata in contexts:
            # Keep some chemically implausible anchor/site combinations out of the search.
            if metadata["site"] == "vacancy" and state_name in {"OH", "H2O", "H2O2"}:
                continue
            if metadata["site"] == "hollow" and state_name in {"H2O2"}:
                continue
            if metadata["site"] == "co_o_bridge" and state_name in {"H2O"}:
                continue
            variants = build_adsorbate_variants(state_name)
            family = f"{ctx_name}__{state_name}"
            for rot_label, ads, anchor_idx in variants:
                atoms = place_adsorbate(slab, ads, anchor_idx, xy, zref, height_by_species[state_name])
                label = f"{state_name}_{rot_label}"
                add_case(
                    candidates,
                    family,
                    label,
                    atoms,
                    info,
                    {
                        **metadata,
                        "state": state_name,
                        "orientation": rot_label,
                        "reaction_neighbors": ";".join(f"{move}:{nxt}" for move, nxt in graph[state_name]),
                    },
                )
    return candidates


def collect_reasonable(summary_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family in sorted({row["family"] for row in summary_rows}):
        fam_rows = [row for row in summary_rows if row["family"] == family]
        best = min(float(row["final_energy_eV"]) for row in fam_rows)
        for row in fam_rows:
            if float(row["final_energy_eV"]) - best > REASONABLE_WINDOW_EV:
                continue
            if float(row["max_force_eVA"]) > REASONABLE_FORCE_EV_A:
                continue
            case_dir = OUT_DIR / row["family"] / row["label"]
            cif_path = case_dir / "relaxed.cif"
            atoms = read(cif_path)
            bad_contacts, min_ratio = has_bad_contacts(atoms)
            if bad_contacts:
                continue
            target = out_dir / f"{row['family']}__{row['label']}"
            target.mkdir(parents=True, exist_ok=True)
            write(target / "structure.cif", atoms)
            record = dict(row)
            record["min_contact_ratio"] = min_ratio
            record["source_cif"] = str(cif_path)
            with open(target / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)
            manifest.append(record)

    manifest.sort(key=lambda row: (row["family"], float(row["deltaE_family_eV"]), row["label"]))
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        cols = [
            "family",
            "label",
            "state",
            "deltaE_family_eV",
            "final_energy_eV",
            "max_force_eVA",
            "orientation",
            "site",
            "co_idx",
            "min_contact_ratio",
            "reaction_neighbors",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in manifest:
            writer.writerow({k: row.get(k, "") for k in cols})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slab = read(INPUT_CIF, format="cif")
    info = identify_surface_sites(slab)
    potential = matgl.load_model(str(MODEL_DIR))
    candidates = build_orr_candidates(slab, info)

    stage1_rows = []
    for idx, cand in enumerate(candidates, start=1):
        case_dir = OUT_DIR / cand["family"] / cand["label"]
        case_dir.mkdir(parents=True, exist_ok=True)
        write(case_dir / "initial.cif", cand["atoms"])
        relaxed_atoms, metrics = relax_candidate(cand["atoms"], potential, EXP_STAGE1_STEPS, EXP_STAGE1_FMAX)
        write(case_dir / "stage1_relaxed.cif", relaxed_atoms)
        row = {
            "case_id": idx,
            "family": cand["family"],
            "label": cand["label"],
            "formula": relaxed_atoms.get_chemical_formula(),
            "natoms": len(relaxed_atoms),
            "energy_per_atom_eV": metrics["final_energy_eV"] / len(relaxed_atoms),
            "stage": "stage1",
            **metrics,
            **cand["metadata"],
        }
        with open(case_dir / "stage1_result.json", "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2, ensure_ascii=False)
        stage1_rows.append(row)

    for family in sorted({row["family"] for row in stage1_rows}):
        fam = [row for row in stage1_rows if row["family"] == family]
        best = min(float(row["final_energy_eV"]) for row in fam)
        for row in fam:
            row["deltaE_family_eV"] = float(row["final_energy_eV"]) - best
    stage1_sorted = sorted(stage1_rows, key=lambda row: (row["family"], row["deltaE_family_eV"], row["final_energy_eV"]))
    write_summary(OUT_DIR / "stage1_summary.csv", stage1_sorted)

    finalists = []
    for family in sorted({row["family"] for row in stage1_sorted}):
        fam = [row for row in stage1_sorted if row["family"] == family]
        finalists.append(fam[0])

    final_rows = []
    for row in finalists:
        case_dir = OUT_DIR / row["family"] / row["label"]
        atoms = read(case_dir / "stage1_relaxed.cif")
        relaxed_atoms, metrics = relax_candidate(atoms, potential, EXP_FINAL_STEPS, EXP_FINAL_FMAX)
        write(case_dir / "relaxed.cif", relaxed_atoms)
        final_row = {
            **row,
            "formula": relaxed_atoms.get_chemical_formula(),
            "natoms": len(relaxed_atoms),
            "energy_per_atom_eV": metrics["final_energy_eV"] / len(relaxed_atoms),
            "stage": "final",
            **metrics,
        }
        with open(case_dir / "result.json", "w", encoding="utf-8") as fh:
            json.dump(final_row, fh, indent=2, ensure_ascii=False)
        final_rows.append(final_row)

    for family in sorted({row["family"] for row in final_rows}):
        fam = [row for row in final_rows if row["family"] == family]
        best = min(float(row["final_energy_eV"]) for row in fam)
        for row in fam:
            row["deltaE_family_eV"] = float(row["final_energy_eV"]) - best
    final_sorted = sorted(final_rows, key=lambda row: (row["family"], row["deltaE_family_eV"], row["final_energy_eV"]))
    write_summary(OUT_DIR / "summary.csv", final_sorted)

    collect_reasonable(final_sorted, OUT_DIR / "reasonable_structures")
    print(f"Wrote {len(stage1_sorted)} stage1 candidates and {len(final_sorted)} final candidates to {OUT_DIR}")


if __name__ == "__main__":
    main()
