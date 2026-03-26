import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE

os.environ.setdefault("MATGL_BACKEND", "DGL")

import matgl
from matgl.ext._ase_dgl import PESCalculator


ROOT = Path(__file__).resolve().parents[1]
INPUT_CIF = ROOT / "intermediates" / "POSCAR.cif"
MODEL_DIR = ROOT / "matgl" / "pretrained_models" / "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"
OUT_DIR = ROOT / "intermediates" / "screening_chgnet_pbe_chemguided"
STAGE1_STEPS = 20
STAGE1_FMAX = 0.20
FINAL_STEPS = 600
FINAL_FMAX = 0.01

A_TOL = 1e-3
AZIMUTHS = tuple(float(x) for x in range(0, 360, 30))
TILTS = (0.0, 30.0)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Zero-length vector cannot be normalized.")
    return vec / norm


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = unit_vector(np.array(axis, dtype=float))
    angle = math.radians(angle_deg)
    ux, uy, uz = axis
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ]
    )


def align_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    v1 = unit_vector(np.array(vec_from, dtype=float))
    v2 = unit_vector(np.array(vec_to, dtype=float))
    cross = np.cross(v1, v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if np.linalg.norm(cross) < 1e-10:
        if dot > 0.999999:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(v1[0]) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = unit_vector(np.cross(v1, trial))
        return rotation_matrix(axis, 180.0)
    axis = unit_vector(cross)
    angle = math.degrees(math.acos(dot))
    return rotation_matrix(axis, angle)


def apply_rotation(ads: Atoms, rot: np.ndarray, center: np.ndarray) -> None:
    positions = ads.get_positions() - center
    ads.set_positions(positions @ rot.T + center)


def canonicalize_adsorbate(species: str, mode: str) -> tuple[Atoms, int]:
    zhat = np.array([0.0, 0.0, 1.0])
    if species == "O":
        return Atoms("O", positions=[[0.0, 0.0, 0.0]]), 0

    if species == "OH":
        ads = molecule("OH")
        syms = ads.get_chemical_symbols()
        o_idx = syms.index("O")
        h_idx = syms.index("H")
        ads.translate(-ads.positions[o_idx])
        oh_vec = ads.positions[h_idx] - ads.positions[o_idx]
        rot = align_vectors(oh_vec, zhat)
        apply_rotation(ads, rot, np.zeros(3))
        return ads, o_idx

    if species == "H2O":
        ads = molecule("H2O")
        syms = ads.get_chemical_symbols()
        o_idx = syms.index("O")
        ads.translate(-ads.positions[o_idx])
        h_indices = [i for i, s in enumerate(syms) if s == "H"]
        bisector = ads.positions[h_indices[0]] + ads.positions[h_indices[1]]
        rot = align_vectors(bisector, zhat)
        apply_rotation(ads, rot, np.zeros(3))
        return ads, o_idx

    if species == "O2":
        ads = molecule("O2")
        lower_idx = int(np.argmin(ads.positions[:, 2]))
        upper_idx = 1 - lower_idx
        ads.translate(-ads.positions[lower_idx])
        bond = ads.positions[upper_idx] - ads.positions[lower_idx]
        if mode == "end_on":
            rot = align_vectors(bond, zhat)
            apply_rotation(ads, rot, np.zeros(3))
            return ads, lower_idx
        if mode == "side_on":
            rot = align_vectors(bond, np.array([1.0, 0.0, 0.0]))
            apply_rotation(ads, rot, np.zeros(3))
            ads.translate([0.0, 0.0, -np.mean(ads.positions[:, 2])])
            return ads, lower_idx
        raise ValueError(f"Unsupported O2 mode: {mode}")

    raise ValueError(f"Unsupported species: {species}")


def build_rotated_adsorbates(species: str, mode: str, tilts: tuple[float, ...], azimuths: tuple[float, ...]) -> list[tuple[str, Atoms, int]]:
    base, anchor_idx = canonicalize_adsorbate(species, mode)
    variants: list[tuple[str, Atoms, int]] = []
    for tilt in tilts:
        curr_azimuths = (0.0,) if abs(tilt) < 1e-12 else azimuths
        for azimuth in curr_azimuths:
            ads = base.copy()
            if tilt:
                apply_rotation(ads, rotation_matrix(np.array([1.0, 0.0, 0.0]), tilt), np.zeros(3))
            if azimuth:
                apply_rotation(ads, rotation_matrix(np.array([0.0, 0.0, 1.0]), azimuth), np.zeros(3))
            label = f"{mode}_tilt{int(tilt)}_az{int(azimuth)}"
            variants.append((label, ads, anchor_idx))
    return variants


def place_adsorbate(
    slab: Atoms,
    ads: Atoms,
    anchor_idx: int,
    anchor_xy: np.ndarray,
    anchor_z: float,
    height: float,
) -> Atoms:
    ads = ads.copy()
    anchor = ads.positions[anchor_idx]
    shift = np.array([anchor_xy[0] - anchor[0], anchor_xy[1] - anchor[1], anchor_z + height - anchor[2]])
    ads.translate(shift)
    combined = slab.copy()
    combined.extend(ads)
    return combined


def pbc_xy_distance(xy1: np.ndarray, xy2: np.ndarray, cell: np.ndarray) -> float:
    delta = xy1 - xy2
    lx, ly = cell[0, 0], cell[1, 1]
    delta[0] -= round(delta[0] / lx) * lx
    delta[1] -= round(delta[1] / ly) * ly
    return float(np.linalg.norm(delta))


def identify_surface_sites(slab: Atoms) -> dict:
    positions = slab.get_positions()
    species = np.array(slab.get_chemical_symbols())
    top_z = positions[:, 2].max()
    top_mask = np.isclose(positions[:, 2], top_z, atol=A_TOL)
    top_o = [int(i) for i in np.where(top_mask & (species == "O"))[0]]
    top_co = [int(i) for i in np.where(top_mask & (species == "Co"))[0]]
    top_o_sorted = sorted(top_o, key=lambda i: (positions[i, 1], positions[i, 0]))
    vacancy_idx = top_o_sorted[0]
    vacancy_xy = positions[vacancy_idx, :2].copy()
    vacancy_z = positions[vacancy_idx, 2]
    top_co_by_dist = sorted(top_co, key=lambda i: pbc_xy_distance(positions[i, :2], vacancy_xy, slab.cell.array))
    neighbor_co = [i for i in top_co_by_dist if pbc_xy_distance(positions[i, :2], vacancy_xy, slab.cell.array) < 2.2 + 1e-6]
    intact_co_site = top_co_by_dist[0]
    hollow_xy = np.array([slab.cell[0, 0] / 4.0, slab.cell[1, 1] / 4.0])
    return {
        "top_z": top_z,
        "vacancy_idx": vacancy_idx,
        "vacancy_xy": vacancy_xy,
        "vacancy_z": vacancy_z,
        "neighbor_co": neighbor_co,
        "intact_co_site": intact_co_site,
        "hollow_xy": hollow_xy,
    }


def freeze_bottom_half(atoms: Atoms, top_z: float) -> None:
    mask = atoms.get_positions()[:, 2] < top_z * 0.5
    atoms.set_constraint(FixAtoms(mask=mask))


def add_case(candidates: list[dict], family: str, label: str, atoms: Atoms, info: dict, metadata: dict) -> None:
    freeze_bottom_half(atoms, info["top_z"])
    candidates.append({"family": family, "label": label, "atoms": atoms, "metadata": metadata})


def add_rotational_family(
    candidates: list[dict],
    slab: Atoms,
    family: str,
    prefix: str,
    species: str,
    mode: str,
    anchor_xy: np.ndarray,
    anchor_z: float,
    height: float,
    tilts: tuple[float, ...],
    azimuths: tuple[float, ...],
    info: dict,
    metadata: dict,
) -> None:
    for rot_label, ads, anchor_idx in build_rotated_adsorbates(species, mode, tilts, azimuths):
        atoms = place_adsorbate(slab, ads, anchor_idx, anchor_xy, anchor_z, height)
        label = f"{prefix}_{rot_label}"
        add_case(candidates, family, label, atoms, info, {**metadata, "orientation": rot_label})


def build_candidates(base_slab: Atoms, info: dict) -> list[dict]:
    candidates: list[dict] = []
    slab_vac = base_slab.copy()
    del slab_vac[info["vacancy_idx"]]
    add_case(candidates, "01_vacancy", "vacancy_only", slab_vac, info, {"vacancy_site": info["vacancy_idx"]})

    co_idx = info["neighbor_co"][0]
    neighbor_xy = base_slab.positions[co_idx, :2]
    co_top_xy = base_slab.positions[info["intact_co_site"], :2]

    add_rotational_family(
        candidates, slab_vac, "02_vacancy_plus_O2", "vac_O2", "O2", "end_on",
        info["vacancy_xy"], info["vacancy_z"], 1.65, TILTS, AZIMUTHS, info, {"site": "vacancy"}
    )
    add_rotational_family(
        candidates, slab_vac, "03_vacancy_plus_O_on_Co", "vac_O_on_Co", "O", "atom",
        neighbor_xy, info["top_z"], 1.55, (0.0,), (0.0,), info, {"site": "neighbor_co_top", "co_idx": co_idx}
    )
    add_rotational_family(
        candidates, slab_vac, "04_vacancy_plus_OH_on_Co", "vac_OH_on_Co", "OH", "co_bound",
        neighbor_xy, info["top_z"], 1.75, TILTS, AZIMUTHS, info, {"site": "neighbor_co_top", "co_idx": co_idx}
    )
    add_rotational_family(
        candidates, slab_vac, "05_vacancy_plus_H2O_on_Co", "vac_H2O_on_Co", "H2O", "co_bound",
        neighbor_xy, info["top_z"], 1.95, TILTS, AZIMUTHS, info, {"site": "neighbor_co_top", "co_idx": co_idx}
    )

    for site_name, xy in (("co_top", co_top_xy), ("hollow", info["hollow_xy"])):
        add_rotational_family(
            candidates, base_slab, "06_direct_O2", f"direct_O2_{site_name}", "O2", "end_on",
            xy, info["top_z"], 1.75, TILTS, AZIMUTHS, info, {"site": site_name}
        )
        add_rotational_family(
            candidates, base_slab, "07_direct_O", f"direct_O_{site_name}", "O", "atom",
            xy, info["top_z"], 1.60 if site_name == "co_top" else 1.30, (0.0,), (0.0,), info, {"site": site_name}
        )
        add_rotational_family(
            candidates, base_slab, "08_direct_OH", f"direct_OH_{site_name}", "OH", "co_bound",
            xy, info["top_z"], 1.80 if site_name == "co_top" else 1.45, TILTS, AZIMUTHS, info, {"site": site_name}
        )
        add_rotational_family(
            candidates, base_slab, "09_direct_H2O", f"direct_H2O_{site_name}", "H2O", "co_bound",
            xy, info["top_z"], 2.00 if site_name == "co_top" else 1.65, TILTS, AZIMUTHS, info, {"site": site_name}
        )
    return candidates


def canonicalize_adsorbate(species: str, mode: str) -> tuple[Atoms, int]:
    if species == "O":
        return Atoms("O", positions=[[0.0, 0.0, 0.0]]), 0
    if species == "OH":
        return canonicalize_oh()
    if species == "H2O":
        return canonicalize_h2o()
    if species == "O2":
        return canonicalize_o2(mode)
    raise ValueError(f"Unsupported species: {species}")


def canonicalize_oh() -> tuple[Atoms, int]:
    ads = molecule("OH")
    syms = ads.get_chemical_symbols()
    o_idx = syms.index("O")
    h_idx = syms.index("H")
    ads.translate(-ads.positions[o_idx])
    oh_vec = ads.positions[h_idx]
    rot = align_vectors(oh_vec, np.array([0.0, 0.0, 1.0]))
    apply_rotation(ads, rot, np.zeros(3))
    return ads, o_idx


def canonicalize_h2o() -> tuple[Atoms, int]:
    ads = molecule("H2O")
    syms = ads.get_chemical_symbols()
    o_idx = syms.index("O")
    h_indices = [i for i, s in enumerate(syms) if s == "H"]
    ads.translate(-ads.positions[o_idx])
    bisector = ads.positions[h_indices[0]] + ads.positions[h_indices[1]]
    rot = align_vectors(bisector, np.array([0.0, 0.0, 1.0]))
    apply_rotation(ads, rot, np.zeros(3))
    return ads, o_idx


def canonicalize_o2(mode: str) -> tuple[Atoms, int]:
    ads = molecule("O2")
    lower_idx = int(np.argmin(ads.positions[:, 2]))
    upper_idx = 1 - lower_idx
    ads.translate(-ads.positions[lower_idx])
    bond = ads.positions[upper_idx]
    target = np.array([0.0, 0.0, 1.0]) if mode == "end_on" else np.array([1.0, 0.0, 0.0])
    rot = align_vectors(bond, target)
    apply_rotation(ads, rot, np.zeros(3))
    return ads, lower_idx


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


def write_summary(path: Path, rows: list[dict]) -> None:
    columns = [
        "case_id", "family", "label", "formula", "natoms", "initial_energy_eV", "final_energy_eV",
        "deltaE_family_eV", "energy_per_atom_eV", "max_force_eVA", "n_steps", "converged",
        "stage", "site", "co_idx", "orientation", "vacancy_site",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slab = read(INPUT_CIF, format="cif")
    info = identify_surface_sites(slab)
    potential = matgl.load_model(str(MODEL_DIR))
    candidates = build_candidates(slab, info)

    stage1_summary = []
    for idx, cand in enumerate(candidates, start=1):
        case_dir = OUT_DIR / cand["family"] / cand["label"]
        case_dir.mkdir(parents=True, exist_ok=True)
        write(case_dir / "initial.cif", cand["atoms"])
        relaxed_atoms, metrics = relax_candidate(cand["atoms"], potential, STAGE1_STEPS, STAGE1_FMAX)
        write(case_dir / "stage1_relaxed.cif", relaxed_atoms)
        record = {
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
            json.dump(record, fh, indent=2, ensure_ascii=False)
        stage1_summary.append(record)

    for family in sorted({row["family"] for row in stage1_summary}):
        family_rows = [row for row in stage1_summary if row["family"] == family]
        best = min(row["final_energy_eV"] for row in family_rows)
        for row in family_rows:
            row["deltaE_family_eV"] = row["final_energy_eV"] - best

    stage1_sorted = sorted(stage1_summary, key=lambda row: (row["family"], row["deltaE_family_eV"], row["final_energy_eV"]))
    write_summary(OUT_DIR / "stage1_summary.csv", stage1_sorted)

    finalists = []
    for family in sorted({row["family"] for row in stage1_sorted}):
        family_rows = [row for row in stage1_sorted if row["family"] == family]
        finalists.append(family_rows[0])

    final_summary = []
    for row in finalists:
        case_dir = OUT_DIR / row["family"] / row["label"]
        atoms = read(case_dir / "stage1_relaxed.cif")
        relaxed_atoms, metrics = relax_candidate(atoms, potential, FINAL_STEPS, FINAL_FMAX)
        write(case_dir / "relaxed.cif", relaxed_atoms)
        final_record = {
            **row,
            "formula": relaxed_atoms.get_chemical_formula(),
            "natoms": len(relaxed_atoms),
            "energy_per_atom_eV": metrics["final_energy_eV"] / len(relaxed_atoms),
            "stage": "final",
            **metrics,
        }
        with open(case_dir / "result.json", "w", encoding="utf-8") as fh:
            json.dump(final_record, fh, indent=2, ensure_ascii=False)
        final_summary.append(final_record)

    for family in sorted({row["family"] for row in final_summary}):
        family_rows = [row for row in final_summary if row["family"] == family]
        best = min(row["final_energy_eV"] for row in family_rows)
        for row in family_rows:
            row["deltaE_family_eV"] = row["final_energy_eV"] - best

    final_sorted = sorted(final_summary, key=lambda row: (row["family"], row["deltaE_family_eV"], row["final_energy_eV"]))
    write_summary(OUT_DIR / "summary.csv", final_sorted)

    top_by_family = {}
    for row in final_sorted:
        top_by_family.setdefault(row["family"], row)
    with open(OUT_DIR / "top_by_family.json", "w", encoding="utf-8") as fh:
        json.dump(top_by_family, fh, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(stage1_sorted)} stage1 candidates and {len(final_sorted)} final refined candidates to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
