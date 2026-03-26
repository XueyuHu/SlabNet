from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

from . import REPO_ROOT


SURFACE_ENERGY_J_PER_M2_PER_EV_PER_A2 = 16.02176634
DEFAULT_MODEL_DIR = REPO_ROOT / "matgl" / "pretrained_models" / "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"


def species_symbol(site) -> str:
    specie = site.specie
    if hasattr(specie, "symbol"):
        return specie.symbol
    if hasattr(specie, "element") and hasattr(specie.element, "symbol"):
        return specie.element.symbol
    return str(specie)


def format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    return "-".join(f"{key}{counter[key]}" for key in sorted(counter))


def termination_counters(slab: Slab, frac_tol: float = 0.08) -> tuple[Counter, Counter]:
    z = np.array([site.frac_coords[2] for site in slab])
    zmax = float(z.max())
    zmin = float(z.min())
    top = Counter(species_symbol(site) for site in slab if zmax - float(site.frac_coords[2]) <= frac_tol)
    bottom = Counter(species_symbol(site) for site in slab if float(site.frac_coords[2]) - zmin <= frac_tol)
    return top, bottom


def freeze_bottom_half(atoms: Atoms) -> list[int]:
    z = atoms.get_positions()[:, 2]
    zmid = 0.5 * (float(z.min()) + float(z.max()))
    fixed = [i for i, zi in enumerate(z) if zi < zmid]
    atoms.set_constraint(FixAtoms(indices=fixed))
    return fixed


def inplane_metrics_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    a_vec = np.array(matrix[0], dtype=float)
    b_vec = np.array(matrix[1], dtype=float)
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    area = float(np.linalg.norm(np.cross(a_vec, b_vec)))
    return a_len, b_len, area


def choose_inplane_repeats(slab: Slab, min_lateral_size: float, min_surface_area: float, max_repeat: int) -> tuple[int, int]:
    a_len, b_len, area = inplane_metrics_from_matrix(np.array(slab.lattice.matrix))
    best: tuple[int, int] | None = None
    best_score: tuple[int, int, float] | None = None
    for na in range(1, max_repeat + 1):
        for nb in range(1, max_repeat + 1):
            if na * a_len < min_lateral_size:
                continue
            if nb * b_len < min_lateral_size:
                continue
            if na * nb * area < min_surface_area:
                continue
            score = (na * nb, max(na, nb), na * nb * area)
            if best_score is None or score < best_score:
                best_score = score
                best = (na, nb)
    if best is None:
        raise ValueError(
            f"Unable to satisfy min_lateral_size={min_lateral_size} A and "
            f"min_surface_area={min_surface_area} A^2 within max_repeat={max_repeat}."
        )
    return best


def expand_slab_inplane(slab: Slab, na: int, nb: int) -> Slab:
    out = slab.copy()
    out.make_supercell([[na, 0, 0], [0, nb, 0], [0, 0, 1]])
    return out


def add_local_matgl_path() -> None:
    src = REPO_ROOT / "matgl" / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    os.environ.setdefault("MATGL_BACKEND", "DGL")


def load_pes_calculator(model_dir: Path):
    add_local_matgl_path()
    import matgl
    from matgl.ext._ase_dgl import PESCalculator

    potential = matgl.load_model(str(model_dir))
    return potential, PESCalculator


def relax_atoms(atoms: Atoms, potential, calculator_cls, fmax: float, steps: int) -> tuple[Atoms, dict]:
    work = deepcopy(atoms)
    work.calc = calculator_cls(potential=potential, stress_unit="eV/A3")
    initial_energy = float(work.get_potential_energy())
    opt = FIRE(work, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    final_energy = float(work.get_potential_energy())
    forces = work.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())
    return work, {
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "n_steps": int(getattr(opt, "nsteps", steps)),
        "max_force_eVA": max_force,
        "converged": bool(max_force <= fmax + 1e-12),
    }


def composition_reference_factor(structure: Structure) -> tuple[str, float]:
    reduced_comp, factor = structure.composition.get_reduced_composition_and_factor()
    return reduced_comp.reduced_formula, float(factor)


def score_surface_library(
    bulk_structure: Structure,
    slab_records: list[dict],
    output_dir: Path,
    model_dir: Path,
    slab_relax_fmax: float,
    slab_relax_steps: int,
    bulk_relax_fmax: float,
    bulk_relax_steps: int,
) -> list[dict]:
    potential, calculator_cls = load_pes_calculator(model_dir)

    bulk_atoms = AseAtomsAdaptor.get_atoms(bulk_structure)
    bulk_relaxed, bulk_metrics = relax_atoms(bulk_atoms, potential, calculator_cls, bulk_relax_fmax, bulk_relax_steps)
    bulk_formula, bulk_factor = composition_reference_factor(bulk_structure)
    bulk_energy_per_fu = bulk_metrics["final_energy_eV"] / bulk_factor

    write(output_dir / "bulk_relaxed.cif", bulk_relaxed)
    bulk_reference = {
        "bulk_formula": bulk_formula,
        "bulk_formula_factor": bulk_factor,
        "bulk_energy_eV": bulk_metrics["final_energy_eV"],
        "bulk_energy_per_formula_unit_eV": bulk_energy_per_fu,
        **bulk_metrics,
        "model_dir": str(model_dir),
    }
    with open(output_dir / "bulk_reference.json", "w", encoding="utf-8") as fh:
        json.dump(bulk_reference, fh, indent=2, ensure_ascii=False)

    scored = []
    for record in slab_records:
        slab_dir = output_dir / record["output_dir"]
        slab_atoms = AseAtomsAdaptor.get_atoms(record["_slab"])
        fixed_indices = freeze_bottom_half(slab_atoms)
        relaxed_slab, slab_metrics = relax_atoms(slab_atoms, potential, calculator_cls, slab_relax_fmax, slab_relax_steps)

        write(slab_dir / "slab_relaxed.cif", relaxed_slab)
        with open(slab_dir / "energy.json", "w", encoding="utf-8") as fh:
            json.dump(
                {**{k: v for k, v in record.items() if not k.startswith("_")}, **slab_metrics, "n_fixed_bottom_half": len(fixed_indices)},
                fh,
                indent=2,
                ensure_ascii=False,
            )

        slab_formula, slab_factor = composition_reference_factor(record["_slab"])
        gamma_eva2 = None
        gamma_jm2 = None
        n_formula_units = None
        if slab_formula == bulk_formula:
            n_formula_units = slab_factor / bulk_factor
            gamma_eva2 = (slab_metrics["final_energy_eV"] - n_formula_units * bulk_energy_per_fu) / (2.0 * record["surface_area_A2"])
            gamma_jm2 = gamma_eva2 * SURFACE_ENERGY_J_PER_M2_PER_EV_PER_A2

        new_record = {
            **record,
            "n_fixed_bottom_half": len(fixed_indices),
            "bulk_reference_formula": bulk_formula,
            "bulk_energy_per_formula_unit_eV": bulk_energy_per_fu,
            "n_formula_units_vs_bulk": n_formula_units,
            "slab_initial_energy_eV": slab_metrics["initial_energy_eV"],
            "slab_final_energy_eV": slab_metrics["final_energy_eV"],
            "slab_relax_steps": slab_metrics["n_steps"],
            "slab_max_force_eVA": slab_metrics["max_force_eVA"],
            "slab_converged": slab_metrics["converged"],
            "surface_energy_eV_A2": gamma_eva2,
            "surface_energy_J_m2": gamma_jm2,
            "relaxed_cif_file": f"{record['output_dir']}/slab_relaxed.cif",
        }
        scored.append(new_record)

    ranked = [row for row in scored if row["surface_energy_J_m2"] is not None]
    ranked.sort(key=lambda row: row["surface_energy_J_m2"])
    for rank, row in enumerate(ranked, start=1):
        row["surface_energy_rank"] = rank
    for row in scored:
        row.setdefault("surface_energy_rank", "")
    return scored


def slab_record(
    slab: Slab,
    slab_id: str,
    output_dir: Path,
    repeat_a: int,
    repeat_b: int,
    min_lateral_size: float,
    min_surface_area: float,
) -> dict:
    top, bottom = termination_counters(slab)
    atoms = AseAtomsAdaptor.get_atoms(slab)
    fixed_indices = freeze_bottom_half(atoms)
    a_len, b_len, area = inplane_metrics_from_matrix(np.array(slab.lattice.matrix))

    slab_dir = output_dir / slab_id
    slab_dir.mkdir(parents=True, exist_ok=True)

    write(slab_dir / "slab.cif", atoms)
    write(slab_dir / "slab_relax_ready.vasp", atoms, format="vasp", direct=True, vasp5=True, sort=False)

    scale = np.array(slab.scale_factor)
    if scale.ndim == 1:
        scale_str = "x".join(str(int(x)) for x in scale)
    else:
        scale_str = ";".join(",".join(str(int(x)) for x in row) for row in scale)

    record = {
        "slab_id": slab_id,
        "miller_index": ",".join(str(x) for x in slab.miller_index),
        "shift": float(slab.shift),
        "formula": slab.composition.reduced_formula,
        "natoms": len(slab),
        "surface_a_A": a_len,
        "surface_b_A": b_len,
        "surface_area_A2": area,
        "repeat_a": repeat_a,
        "repeat_b": repeat_b,
        "min_lateral_target_A": min_lateral_size,
        "min_surface_area_target_A2": min_surface_area,
        "slab_scale_factor": scale_str,
        "top_termination": format_counter(top),
        "bottom_termination": format_counter(bottom),
        "is_symmetric": bool(top == bottom),
        "is_polar": bool(slab.is_polar()),
        "n_fixed_bottom_half": len(fixed_indices),
        "output_dir": slab_id,
        "cif_file": f"{slab_id}/slab.cif",
        "relax_ready_vasp": f"{slab_id}/slab_relax_ready.vasp",
        "_slab": slab,
    }

    with open(slab_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in record.items() if not k.startswith("_")}, fh, indent=2, ensure_ascii=False)

    return record


def write_summary(records: list[dict], output_dir: Path) -> None:
    out_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in records]
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        cols = [
            "slab_id",
            "miller_index",
            "shift",
            "formula",
            "natoms",
            "surface_a_A",
            "surface_b_A",
            "surface_area_A2",
            "repeat_a",
            "repeat_b",
            "min_lateral_target_A",
            "min_surface_area_target_A2",
            "slab_scale_factor",
            "top_termination",
            "bottom_termination",
            "is_symmetric",
            "is_polar",
            "n_fixed_bottom_half",
            "bulk_reference_formula",
            "bulk_energy_per_formula_unit_eV",
            "n_formula_units_vs_bulk",
            "slab_initial_energy_eV",
            "slab_final_energy_eV",
            "slab_relax_steps",
            "slab_max_force_eVA",
            "slab_converged",
            "surface_energy_eV_A2",
            "surface_energy_J_m2",
            "surface_energy_rank",
            "output_dir",
            "cif_file",
            "relax_ready_vasp",
            "relaxed_cif_file",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({col: row.get(col, "") for col in cols})

    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(out_rows, fh, indent=2, ensure_ascii=False)


def generate_surface_library(
    input_path: Path,
    output_dir: Path,
    max_index: int = 2,
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    center_slab: bool = True,
    primitive: bool = False,
    max_normal_search: int | None = None,
    min_lateral_size: float = 8.0,
    min_surface_area: float = 64.0,
    max_repeat: int = 6,
    score_model: bool = False,
    model_dir: Path | None = None,
    slab_relax_fmax: float = 0.03,
    slab_relax_steps: int = 400,
    bulk_relax_fmax: float = 0.01,
    bulk_relax_steps: int = 500,
) -> list[dict]:
    structure = Structure.from_file(str(input_path))
    slabs = generate_all_slabs(
        structure,
        max_index=max_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
        primitive=primitive,
        max_normal_search=max_normal_search,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    hkl_counts: dict[tuple[int, int, int], int] = {}
    for slab in slabs:
        hkl = tuple(int(x) for x in slab.miller_index)
        hkl_counts[hkl] = hkl_counts.get(hkl, 0) + 1
        term_no = hkl_counts[hkl]
        na, nb = choose_inplane_repeats(slab, min_lateral_size, min_surface_area, max_repeat)
        slab = expand_slab_inplane(slab, na, nb)
        hkl_label = "".join(str(x) for x in hkl)
        top, bottom = termination_counters(slab)
        slab_id = (
            f"hkl_{hkl_label}"
            f"__term_{term_no:02d}"
            f"__rep_{na}x{nb}"
            f"__top_{format_counter(top)}"
            f"__bot_{format_counter(bottom)}"
        )
        records.append(slab_record(slab, slab_id, output_dir, na, nb, min_lateral_size, min_surface_area))

    if score_model:
        records = score_surface_library(
            bulk_structure=structure,
            slab_records=records,
            output_dir=output_dir,
            model_dir=model_dir or DEFAULT_MODEL_DIR,
            slab_relax_fmax=slab_relax_fmax,
            slab_relax_steps=slab_relax_steps,
            bulk_relax_fmax=bulk_relax_fmax,
            bulk_relax_steps=bulk_relax_steps,
        )

    records.sort(
        key=lambda row: (
            math.inf if row.get("surface_energy_rank", "") == "" else int(row["surface_energy_rank"]),
            tuple(int(x) for x in row["miller_index"].split(",")),
            row["shift"],
            row["slab_id"],
        )
    )
    write_summary(records, output_dir)
    return records
