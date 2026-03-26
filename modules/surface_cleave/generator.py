from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import write
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor


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


def freeze_bottom_half(atoms):
    z = atoms.get_positions()[:, 2]
    zmid = 0.5 * (float(z.min()) + float(z.max()))
    fixed = [i for i, zi in enumerate(z) if zi < zmid]
    atoms.set_constraint(FixAtoms(indices=fixed))
    return fixed


def slab_record(slab: Slab, slab_id: str, output_dir: Path) -> dict:
    top, bottom = termination_counters(slab)
    atoms = AseAtomsAdaptor.get_atoms(slab)
    fixed_indices = freeze_bottom_half(atoms)
    area = float(np.linalg.norm(np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])))

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
        "surface_area_A2": area,
        "slab_scale_factor": scale_str,
        "top_termination": format_counter(top),
        "bottom_termination": format_counter(bottom),
        "is_symmetric": bool(top == bottom),
        "is_polar": bool(slab.is_polar()),
        "n_fixed_bottom_half": len(fixed_indices),
        "output_dir": slab_id,
        "cif_file": f"{slab_id}/slab.cif",
        "relax_ready_vasp": f"{slab_id}/slab_relax_ready.vasp",
    }

    with open(slab_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    return record


def generate_surface_library(
    input_path: Path,
    output_dir: Path,
    max_index: int = 2,
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    center_slab: bool = True,
    primitive: bool = False,
    max_normal_search: int | None = None,
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
        hkl_label = "".join(str(x) for x in hkl)
        top, bottom = termination_counters(slab)
        slab_id = (
            f"hkl_{hkl_label}"
            f"__term_{term_no:02d}"
            f"__top_{format_counter(top)}"
            f"__bot_{format_counter(bottom)}"
        )
        records.append(slab_record(slab, slab_id, output_dir))

    records.sort(key=lambda row: (tuple(int(x) for x in row["miller_index"].split(",")), row["shift"], row["slab_id"]))
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        cols = [
            "slab_id",
            "miller_index",
            "shift",
            "formula",
            "natoms",
            "surface_area_A2",
            "slab_scale_factor",
            "top_termination",
            "bottom_termination",
            "is_symmetric",
            "is_polar",
            "n_fixed_bottom_half",
            "output_dir",
            "cif_file",
            "relax_ready_vasp",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    return records
