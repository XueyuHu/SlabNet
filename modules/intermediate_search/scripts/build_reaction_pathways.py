import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "intermediates" / "orr_search_chgnet_pbe" / "reasonable_structures" / "manifest.csv"
OUT_DIR = ROOT / "intermediates" / "orr_search_chgnet_pbe" / "reaction_network"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def node_id(row: dict) -> str:
    return f"{row['family']}__{row['label']}"


def parse_neighbors(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in raw.split(";"):
        if not chunk:
            continue
        move, target = chunk.split(":")
        pairs.append((move, target))
    return pairs


def build_graph(rows: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    nodes = {node_id(row): row for row in rows}
    by_context_state = {}
    for row in rows:
        context = row["family"].rsplit("__", 1)[0]
        by_context_state[(context, row["state"])] = row

    edges = []
    for row in rows:
        src = node_id(row)
        context = row["family"].rsplit("__", 1)[0]
        for move, target_state in parse_neighbors(row["reaction_neighbors"]):
            target = by_context_state.get((context, target_state))
            if target is None:
                continue
            edges.append(
                {
                    "source": src,
                    "target": node_id(target),
                    "move": move,
                    "context": context,
                    "source_state": row["state"],
                    "target_state": target_state,
                }
            )
    return nodes, edges


def adjacency(edges: list[dict]) -> dict[str, list[dict]]:
    graph = {}
    for edge in edges:
        graph.setdefault(edge["source"], []).append(edge)
    return graph


def find_all_simple_paths(graph: dict[str, list[dict]], start: str, goal: str, max_depth: int = 6) -> list[list[dict]]:
    paths = []

    def dfs(current: str, target: str, used: set[str], trail: list[dict]) -> None:
        if len(trail) > max_depth:
            return
        if current == target:
            paths.append(trail.copy())
            return
        for edge in graph.get(current, []):
            nxt = edge["target"]
            if nxt in used:
                continue
            used.add(nxt)
            trail.append(edge)
            dfs(nxt, target, used, trail)
            trail.pop()
            used.remove(nxt)

    dfs(start, goal, {start}, [])
    return paths


def path_states(nodes: dict[str, dict], start: str, edges: list[dict]) -> list[str]:
    states = [nodes[start]["state"]]
    for edge in edges:
        states.append(nodes[edge["target"]]["state"])
    return states


def plot_context_path_counts(context_counts: dict[str, int], out_png: Path) -> None:
    labels = list(context_counts)
    values = [context_counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2a6f97")
    ax.set_ylabel("Complete O2* -> H2O* Paths")
    ax.set_title("Reaction Pathway Counts by Context")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(MANIFEST_CSV)
    nodes, edges = build_graph(rows)
    graph = adjacency(edges)

    with open(OUT_DIR / "nodes.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["node_id", "family", "label", "state", "site", "co_idx", "final_energy_eV", "max_force_eVA"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for nid, row in nodes.items():
            writer.writerow(
                {
                    "node_id": nid,
                    "family": row["family"],
                    "label": row["label"],
                    "state": row["state"],
                    "site": row["site"],
                    "co_idx": row.get("co_idx", ""),
                    "final_energy_eV": row["final_energy_eV"],
                    "max_force_eVA": row["max_force_eVA"],
                }
            )

    with open(OUT_DIR / "edges.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["context", "source", "move", "target", "source_state", "target_state"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(edges)

    by_context = {}
    for nid, row in nodes.items():
        context = row["family"].rsplit("__", 1)[0]
        by_context.setdefault(context, []).append(nid)

    pathway_rows = []
    pathway_text = []
    context_counts = {}
    path_index = 1
    for context, node_ids in sorted(by_context.items()):
        starts = [nid for nid in node_ids if nodes[nid]["state"] == "O2"]
        goals = [nid for nid in node_ids if nodes[nid]["state"] == "H2O"]
        all_paths = []
        for start in starts:
            for goal in goals:
                all_paths.extend(find_all_simple_paths(graph, start, goal))
        context_counts[context] = len(all_paths)
        for edge_path in all_paths:
            start = edge_path[0]["source"]
            states = path_states(nodes, start, edge_path)
            pathway_text.append(f"path_{path_index} [{context}]: " + " -> ".join(states))
            pathway_rows.append(
                {
                    "path_id": f"path_{path_index}",
                    "context": context,
                    "n_steps": len(states),
                    "states": " -> ".join(states),
                    "moves": " -> ".join(edge["move"] for edge in edge_path),
                    "start_node": start,
                    "end_node": edge_path[-1]["target"],
                }
            )
            path_index += 1

    with open(OUT_DIR / "all_paths.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["path_id", "context", "n_steps", "states", "moves", "start_node", "end_node"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(pathway_rows)

    with open(OUT_DIR / "all_paths.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(pathway_text) + "\n")

    with open(OUT_DIR / "context_counts.json", "w", encoding="utf-8") as fh:
        json.dump(context_counts, fh, indent=2, ensure_ascii=False)

    plot_context_path_counts(context_counts, OUT_DIR / "context_path_counts.png")
    print(f"Wrote {len(nodes)} nodes, {len(edges)} edges, and {len(pathway_rows)} complete O2->H2O paths to {OUT_DIR}")


if __name__ == "__main__":
    main()
