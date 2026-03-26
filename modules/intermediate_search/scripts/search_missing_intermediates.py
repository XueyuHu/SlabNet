import csv
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

os.environ.setdefault("MATGL_BACKEND", "DGL")

import matgl
from matgl.ext._ase_dgl import PESCalculator

from ASE.search_orr_intermediates import (
    INPUT_CIF,
    MODEL_DIR,
    OUT_DIR as MAIN_OUT_DIR,
    REASONABLE_FORCE_EV_A,
    SHORT_CONTACT_SCALE,
    build_adsorbate_variants,
    canonicalize_custom,
    covalent_sum,
    has_bad_contacts,
)
from ASE.screen_chgnet_intermediates import AZIMUTHS, TILTS, add_case, identify_surface_sites, place_adsorbate, write_summary


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediates" / "orr_search_chgnet_pbe_completion"
STAGE1_STEPS = 16
STAGE1_FMAX = 0.25
FINAL_STEPS = 900
FINAL_FMAX = 0.01


def relax_candidate(atoms, potential, steps: int, fmax: float):
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


def load_current_reasonable() -> set[tuple[str, str]]:
    manifest = MAIN_OUT_DIR / "reasonable_structures" / "manifest.csv"
    if not manifest.exists():
        return set()
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(row["family"].rsplit("__", 1)[0], row["state"]) for row in rows}


def build_missing_candidates(base_slab, info) -> list[dict]:
    positions = base_slab.get_positions()
    slab_vac = base_slab.copy()
    del slab_vac[info["vacancy_idx"]]

    missing_targets = [
        ("direct_co_o_bridge", "H2O"),
        ("vacancy_center", "OH"),
        ("vacancy_center", "H2O"),
        ("vacancy_center", "H2O2"),
        ("vacancy_co_top_3", "OH"),
        ("vacancy_co_top_3", "H2O2"),
        ("vacancy_co_top_13", "OH"),
        ("vacancy_co_top_13", "H2O2"),
    ]

    direct_co_xy = positions[info["intact_co_site"], :2]
    top_o_idx = min(
        [i for i, s in enumerate(base_slab.get_chemical_symbols()) if s == "O" and abs(positions[i, 2] - info["top_z"]) < 1e-3],
        key=lambda i: np.linalg.norm(positions[i, :2] - direct_co_xy),
    )
    direct_bridge_xy = 0.5 * (direct_co_xy + positions[top_o_idx, :2])

    anchor_map = {
        ("direct_co_o_bridge", "H2O"): [
            ("bridge", base_slab, direct_bridge_xy, info["top_z"], [1.7, 1.9, 2.1]),
        ],
        ("vacancy_center", "OH"): [
            ("center", slab_vac, info["vacancy_xy"], info["vacancy_z"], [1.4, 1.7]),
            ("offset_x+", slab_vac, info["vacancy_xy"] + np.array([0.45, 0.0]), info["top_z"], [1.6]),
            ("offset_y+", slab_vac, info["vacancy_xy"] + np.array([0.0, 0.45]), info["top_z"], [1.6]),
        ],
        ("vacancy_center", "H2O"): [
            ("center", slab_vac, info["vacancy_xy"], info["vacancy_z"], [1.8, 2.0]),
            ("offset_x+", slab_vac, info["vacancy_xy"] + np.array([0.45, 0.0]), info["top_z"], [1.9]),
            ("offset_y+", slab_vac, info["vacancy_xy"] + np.array([0.0, 0.45]), info["top_z"], [1.9]),
        ],
        ("vacancy_center", "H2O2"): [
            ("center", slab_vac, info["vacancy_xy"], info["vacancy_z"], [1.8, 2.0]),
            ("offset_diag", slab_vac, info["vacancy_xy"] + np.array([0.35, 0.35]), info["top_z"], [1.9]),
        ],
        ("vacancy_co_top_3", "OH"): [
            ("cotop3", slab_vac, positions[3, :2], info["top_z"], [1.7, 1.9]),
        ],
        ("vacancy_co_top_3", "H2O2"): [
            ("cotop3", slab_vac, positions[3, :2], info["top_z"], [1.9, 2.1]),
        ],
        ("vacancy_co_top_13", "OH"): [
            ("cotop13", slab_vac, positions[13, :2], info["top_z"], [1.7, 1.9]),
        ],
        ("vacancy_co_top_13", "H2O2"): [
            ("cotop13", slab_vac, positions[13, :2], info["top_z"], [1.9, 2.1]),
        ],
    }

    candidates = []
    for context, state in missing_targets:
        variants = build_adsorbate_variants(state) if state not in {"H2O2"} else build_adsorbate_variants("H2O2")
        for anchor_name, slab, xy, zref, heights in anchor_map[(context, state)]:
            for height in heights:
                for rot_label, ads, anchor_idx in variants:
                    label = f"{state}_{anchor_name}_h{height:.2f}_{rot_label}".replace(".", "p")
                    atoms = place_adsorbate(slab, ads, anchor_idx, xy, zref, height)
                    add_case(
                        candidates,
                        f"{context}__{state}",
                        label,
                        atoms,
                        info,
                        {
                            "site": context,
                            "state": state,
                            "orientation": rot_label,
                            "anchor_variant": anchor_name,
                            "height_A": height,
                        },
                    )
    return candidates


def collect_reasonable(rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for row in rows:
        if float(row["max_force_eVA"]) > REASONABLE_FORCE_EV_A:
            continue
        case_dir = OUT_DIR / row["family"] / row["label"]
        cif_path = case_dir / "relaxed.cif"
        atoms = read(cif_path)
        bad, min_ratio = has_bad_contacts(atoms)
        if bad:
            continue
        target_dir = out_dir / f"{row['family']}__{row['label']}"
        target_dir.mkdir(parents=True, exist_ok=True)
        write(target_dir / "structure.cif", atoms)
        rec = dict(row)
        rec["min_contact_ratio"] = min_ratio
        rec["source_cif"] = str(cif_path)
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        kept.append(rec)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=2, ensure_ascii=False)
    return kept


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slab = read(INPUT_CIF, format="cif")
    info = identify_surface_sites(slab)
    potential = matgl.load_model(str(MODEL_DIR))
    candidates = build_missing_candidates(slab, info)

    stage1 = []
    for idx, cand in enumerate(candidates, start=1):
        case_dir = OUT_DIR / cand["family"] / cand["label"]
        case_dir.mkdir(parents=True, exist_ok=True)
        write(case_dir / "initial.cif", cand["atoms"])
        relaxed_atoms, metrics = relax_candidate(cand["atoms"], potential, STAGE1_STEPS, STAGE1_FMAX)
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
        stage1.append(row)

    for family in sorted({r["family"] for r in stage1}):
        fam = [r for r in stage1 if r["family"] == family]
        best = min(float(r["final_energy_eV"]) for r in fam)
        for r in fam:
            r["deltaE_family_eV"] = float(r["final_energy_eV"]) - best
    stage1_sorted = sorted(stage1, key=lambda r: (r["family"], r["deltaE_family_eV"], r["final_energy_eV"]))
    write_summary(OUT_DIR / "stage1_summary.csv", stage1_sorted)

    finalists = []
    for family in sorted({r["family"] for r in stage1_sorted}):
        fam = [r for r in stage1_sorted if r["family"] == family]
        finalists.extend(fam[:2])

    final_rows = []
    for row in finalists:
        case_dir = OUT_DIR / row["family"] / row["label"]
        atoms = read(case_dir / "stage1_relaxed.cif")
        relaxed_atoms, metrics = relax_candidate(atoms, potential, FINAL_STEPS, FINAL_FMAX)
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

    for family in sorted({r["family"] for r in final_rows}):
        fam = [r for r in final_rows if r["family"] == family]
        best = min(float(r["final_energy_eV"]) for r in fam)
        for r in fam:
            r["deltaE_family_eV"] = float(r["final_energy_eV"]) - best
    final_sorted = sorted(final_rows, key=lambda r: (r["family"], r["deltaE_family_eV"], r["final_energy_eV"]))
    write_summary(OUT_DIR / "summary.csv", final_sorted)

    kept = collect_reasonable(final_sorted, OUT_DIR / "reasonable_structures")
    print(f"Wrote {len(stage1_sorted)} stage1 candidates, {len(final_sorted)} refined candidates, {len(kept)} reasonable completions to {OUT_DIR}")


if __name__ == "__main__":
    main()
