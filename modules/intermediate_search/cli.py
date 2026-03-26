from __future__ import annotations

import argparse
import csv
import json
import runpy
import sys
from pathlib import Path

from . import MODULE_ROOT, REPO_ROOT


RESULTS_DIR = MODULE_ROOT / "results"
MERGED_NETWORK_DIR = RESULTS_DIR / "merged_network"
REASONABLE_DIR = RESULTS_DIR / "reasonable_structures_merged"

SCRIPT_MAP = {
    "screen": REPO_ROOT / "ASE" / "screen_chgnet_intermediates.py",
    "search": REPO_ROOT / "ASE" / "search_orr_intermediates.py",
    "complete": REPO_ROOT / "ASE" / "search_missing_intermediates.py",
    "network": REPO_ROOT / "ASE" / "build_reaction_pathways.py",
    "graph": REPO_ROOT / "ASE" / "draw_reaction_graph.py",
    "profiles": REPO_ROOT / "ASE" / "build_merged_reaction_profiles.py",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def print_overview() -> int:
    path_counts = read_json(MERGED_NETWORK_DIR / "path_counts.json")
    nodes = read_csv(MERGED_NETWORK_DIR / "nodes.csv")
    edges = read_csv(MERGED_NETWORK_DIR / "edges.csv")
    structures = read_csv(REASONABLE_DIR / "manifest.csv")
    contexts = sorted({row["context"] for row in nodes})

    print("SlabNet Intermediate Search")
    print(f"module_root: {MODULE_ROOT}")
    print(f"repo_root:   {REPO_ROOT}")
    print(f"reasonable_structures: {len(structures)}")
    print(f"reaction_nodes:        {len(nodes)}")
    print(f"reaction_edges:        {len(edges)}")
    print(f"pathways_total:        {path_counts['total']}")
    print(f"pathways_complete:     {path_counts['complete']}")
    print(f"pathways_partial:      {path_counts['partial']}")
    print("contexts:              " + ", ".join(contexts))
    return 0


def print_paths(status: str, context: str | None, limit: int | None) -> int:
    rows = read_csv(MERGED_NETWORK_DIR / "all_paths.csv")
    if status != "all":
        rows = [row for row in rows if row["status"] == status]
    if context:
        rows = [row for row in rows if row["context"] == context]
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("No paths matched the requested filters.")
        return 0

    for row in rows:
        print(f"{row['path_id']} [{row['context']}] ({row['status']}): {row['state_labels']}")
    return 0


def print_describe() -> int:
    description = (MERGED_NETWORK_DIR / "picture_description.txt").read_text(encoding="utf-8")
    print(description.rstrip())
    return 0


def print_files() -> int:
    key_files = [
        MODULE_ROOT / "README.md",
        MODULE_ROOT / "docs" / "workflow.md",
        MODULE_ROOT / "inputs" / "POSCAR.cif",
        RESULTS_DIR / "main_search" / "summary.csv",
        RESULTS_DIR / "completion_search" / "summary.csv",
        MERGED_NETWORK_DIR / "all_paths.txt",
        MERGED_NETWORK_DIR / "energy_profiles" / "all_paths_profiles.png",
        REASONABLE_DIR / "manifest.csv",
    ]
    for path in key_files:
        print(path.relative_to(REPO_ROOT))
    return 0


def run_workflow_script(step: str) -> int:
    script = SCRIPT_MAP[step]
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return 1
    runpy.run_path(str(script), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modules.intermediate_search",
        description="CLI for the SlabNet intermediate-search module.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("overview", help="Print a compact summary of the packaged module results.")

    paths_p = sub.add_parser("paths", help="List reaction pathways from the merged network.")
    paths_p.add_argument("--status", choices=["all", "complete", "partial"], default="all")
    paths_p.add_argument("--context", default=None, help="Filter by context, e.g. direct_co_top.")
    paths_p.add_argument("--limit", type=int, default=None, help="Limit the number of paths shown.")

    sub.add_parser("describe", help="Print the packaged picture description for the merged profiles.")
    sub.add_parser("files", help="List the most important packaged files in this module.")

    run_p = sub.add_parser("run", help="Run one step of the original workflow from the repository scripts.")
    run_p.add_argument("step", choices=sorted(SCRIPT_MAP), help="Workflow step to execute.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "overview":
        return print_overview()
    if args.command == "paths":
        return print_paths(args.status, args.context, args.limit)
    if args.command == "describe":
        return print_describe()
    if args.command == "files":
        return print_files()
    if args.command == "run":
        return run_workflow_script(args.step)

    parser.print_help()
    return 1

