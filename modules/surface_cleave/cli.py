from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from . import MODULE_ROOT, REPO_ROOT


INPUT_DEFAULT = REPO_ROOT / "surface_cleave" / "POSCAR.cif"
RESULTS_DIR = MODULE_ROOT / "results" / "generated_slabs"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cmd_generate(args) -> int:
    from .generator import generate_surface_library

    records = generate_surface_library(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        max_index=args.max_index,
        min_slab_size=args.min_slab,
        min_vacuum_size=args.min_vacuum,
        center_slab=not args.no_center,
        primitive=args.primitive,
        max_normal_search=args.max_normal_search,
        min_lateral_size=args.min_lateral_size,
        min_surface_area=args.min_surface_area,
        max_repeat=args.max_repeat,
        score_model=args.score_model,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        slab_relax_fmax=args.slab_relax_fmax,
        slab_relax_steps=args.slab_relax_steps,
        bulk_relax_fmax=args.bulk_relax_fmax,
        bulk_relax_steps=args.bulk_relax_steps,
    )
    print(f"Generated {len(records)} slab terminations in {args.output}")
    return 0


def cmd_overview(args) -> int:
    summary = Path(args.summary)
    if not summary.exists():
        print(f"Summary not found: {summary}")
        return 1
    rows = read_csv(summary)
    by_hkl = Counter(row["miller_index"] for row in rows)
    print("Slab cleave overview")
    print(f"summary:             {summary}")
    print(f"n_terminations:      {len(rows)}")
    print(f"n_unique_hkls:       {len(by_hkl)}")
    if rows and rows[0].get("surface_energy_rank") is not None:
        ranked = [row for row in rows if row.get("surface_energy_rank", "") not in {"", None}]
        if ranked:
            print(f"ranked_surfaces:     {len(ranked)}")
    print("terminations_by_hkl:")
    for hkl, count in sorted(by_hkl.items()):
        print(f"  {hkl}: {count}")
    return 0


def cmd_list(args) -> int:
    summary = Path(args.summary)
    rows = read_csv(summary)
    rows.sort(
        key=lambda row: (
            float("inf") if row.get("surface_energy_rank", "") in {"", None} else int(row["surface_energy_rank"]),
            row["miller_index"],
            row["slab_id"],
        )
    )
    if args.hkl:
        rows = [row for row in rows if row["miller_index"] == args.hkl]
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        print(
            f"{row['slab_id']} | hkl={row['miller_index']} | "
            f"a={float(row['surface_a_A']):.2f}A | b={float(row['surface_b_A']):.2f}A | "
            f"top={row['top_termination']} | bottom={row['bottom_termination']} | "
            f"polar={row['is_polar']} | symmetric={row['is_symmetric']} | "
            f"gamma={row.get('surface_energy_J_m2', '')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modules.surface_cleave",
        description="Generate low-index slab terminations from a bulk structure.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate low-index slabs and save all terminations.")
    gen.add_argument("--input", default=str(INPUT_DEFAULT), help="Bulk structure input file.")
    gen.add_argument("--output", default=str(RESULTS_DIR), help="Output directory for generated slabs.")
    gen.add_argument("--max-index", type=int, default=2, help="Maximum Miller index to enumerate.")
    gen.add_argument("--min-slab", type=float, default=10.0, help="Minimum slab thickness in angstrom.")
    gen.add_argument("--min-vacuum", type=float, default=15.0, help="Minimum vacuum size in angstrom.")
    gen.add_argument("--min-lateral-size", type=float, default=8.0, help="Minimum in-plane lattice length for both slab axes.")
    gen.add_argument("--min-surface-area", type=float, default=64.0, help="Minimum in-plane surface area in angstrom^2.")
    gen.add_argument("--max-repeat", type=int, default=6, help="Maximum repeat allowed when expanding the in-plane slab cell.")
    gen.add_argument("--max-normal-search", type=int, default=None, help="Optional max_normal_search for pymatgen.")
    gen.add_argument("--primitive", action="store_true", help="Use primitive cell cleaving instead of conventional.")
    gen.add_argument("--no-center", action="store_true", help="Do not center the slab in vacuum.")
    gen.add_argument("--score-model", action="store_true", help="Use the CHGNet MatPES PBE model to relax slabs and estimate surface energies.")
    gen.add_argument("--model-dir", default=None, help="Optional path to a matgl PES model directory.")
    gen.add_argument("--slab-relax-fmax", type=float, default=0.03, help="Force threshold for slab scoring relaxations.")
    gen.add_argument("--slab-relax-steps", type=int, default=400, help="Maximum FIRE steps for slab scoring relaxations.")
    gen.add_argument("--bulk-relax-fmax", type=float, default=0.01, help="Force threshold for bulk reference relaxation.")
    gen.add_argument("--bulk-relax-steps", type=int, default=500, help="Maximum FIRE steps for bulk reference relaxation.")
    gen.set_defaults(func=cmd_generate)

    overview = sub.add_parser("overview", help="Show a summary of the generated slab library.")
    overview.add_argument("--summary", default=str(RESULTS_DIR / "summary.csv"))
    overview.set_defaults(func=cmd_overview)

    listing = sub.add_parser("list", help="List generated slab terminations from the summary table.")
    listing.add_argument("--summary", default=str(RESULTS_DIR / "summary.csv"))
    listing.add_argument("--hkl", default=None, help="Filter by Miller index, e.g. 1,0,0")
    listing.add_argument("--limit", type=int, default=None)
    listing.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
