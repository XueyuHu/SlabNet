import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ASE.oh_reaction_search_space import allowed_neighbors


ROOT = Path(__file__).resolve().parents[1]
MAIN_MANIFEST = ROOT / "intermediates" / "orr_search_chgnet_pbe" / "reasonable_structures" / "manifest.csv"
COMPLETION_MANIFEST = ROOT / "intermediates" / "orr_search_chgnet_pbe_completion" / "reasonable_structures" / "manifest.json"
OUT_DIR = ROOT / "intermediates" / "orr_search_chgnet_pbe" / "reaction_network_merged"

STATE_LABELS = {
    "O2": "OO*",
    "O": "O*",
    "OH": "OH*",
    "OOH": "OOH*",
    "H2O2": "H2O2*",
    "H2O": "H2O*",
    "bare": "*",
}

STATE_ORDER = {"O2": 0, "O": 1, "OH": 2, "OOH": 3, "H2O2": 4, "H2O": 5, "bare": 6}
CONTEXT_COLORS = {
    "direct_co_top": "#0b6e4f",
    "direct_hollow": "#c44536",
    "direct_co_o_bridge": "#9c6b30",
    "vacancy_center": "#7a3eb1",
    "vacancy_edge_bridge": "#2b59c3",
    "vacancy_co_top_3": "#b56576",
    "vacancy_co_top_13": "#3a7d44",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def node_id(row: dict) -> str:
    return f"{row['family']}__{row['label']}"


def normalize_row(row: dict) -> dict:
    out = dict(row)
    out["context"] = out["family"].rsplit("__", 1)[0]
    out["node_id"] = node_id(out)
    out["deltaE_family_eV"] = float(out.get("deltaE_family_eV", 0.0))
    out["final_energy_eV"] = float(out["final_energy_eV"])
    out["max_force_eVA"] = float(out["max_force_eVA"])
    out["min_contact_ratio"] = float(out.get("min_contact_ratio", 1.0))
    raw_neighbors = out.get("reaction_neighbors", "")
    if not raw_neighbors:
        out["reaction_neighbors"] = ";".join(f"{move}:{nxt}" for move, nxt in allowed_neighbors(out["state"]))
    return out


def load_rows() -> list[dict]:
    rows = [normalize_row(row) for row in read_csv(MAIN_MANIFEST)]
    if COMPLETION_MANIFEST.exists():
        rows.extend(normalize_row(row) for row in read_json(COMPLETION_MANIFEST))
    rows.sort(key=lambda row: (row["context"], STATE_ORDER.get(row["state"], 99), row["deltaE_family_eV"], row["label"]))
    return rows


def parse_neighbors(raw: str) -> list[tuple[str, str]]:
    neighbors = []
    for chunk in raw.split(";"):
        if not chunk:
            continue
        move, state = chunk.split(":")
        neighbors.append((move, state))
    return neighbors


def build_graph(rows: list[dict]) -> tuple[dict[str, dict], list[dict], dict[str, list[str]]]:
    nodes = {row["node_id"]: row for row in rows}
    by_context_state = defaultdict(list)
    for row in rows:
        by_context_state[f"{row['context']}::{row['state']}"].append(row["node_id"])

    edges = []
    adjacency = defaultdict(list)
    for row in rows:
        src = row["node_id"]
        for move, target_state in parse_neighbors(row["reaction_neighbors"]):
            targets = by_context_state.get(f"{row['context']}::{target_state}", [])
            for target in targets:
                edge = {
                    "context": row["context"],
                    "source": src,
                    "target": target,
                    "move": move,
                    "source_state": row["state"],
                    "target_state": target_state,
                }
                edges.append(edge)
                adjacency[src].append(target)
    return nodes, edges, adjacency


def enumerate_maximal_paths(nodes: dict[str, dict], edges: list[dict]) -> list[dict]:
    edges_by_source = defaultdict(list)
    for edge in edges:
        edges_by_source[edge["source"]].append(edge)

    paths = []

    def dfs(current: str, trail: list[dict], used_nodes: set[str], used_states: set[str]) -> None:
        forward = [
            edge
            for edge in edges_by_source.get(current, [])
            if edge["target"] not in used_nodes and nodes[edge["target"]]["state"] not in used_states
        ]
        if not forward:
            path_nodes = [trail[0]["source"]] + [edge["target"] for edge in trail] if trail else [current]
            paths.append(
                {
                    "context": nodes[current]["context"] if not trail else trail[0]["context"],
                    "node_ids": path_nodes,
                    "states": [nodes[nid]["state"] for nid in path_nodes],
                    "moves": [edge["move"] for edge in trail],
                }
            )
            return
        for edge in forward:
            target_state = nodes[edge["target"]]["state"]
            used_nodes.add(edge["target"])
            used_states.add(target_state)
            trail.append(edge)
            dfs(edge["target"], trail, used_nodes, used_states)
            trail.pop()
            used_nodes.remove(edge["target"])
            used_states.remove(target_state)

    start_nodes = [nid for nid, row in nodes.items() if row["state"] == "O2"]
    for start in start_nodes:
        dfs(start, [], {start}, {nodes[start]["state"]})

    dedup = {}
    for path in paths:
        key = (path["context"], tuple(path["node_ids"]))
        dedup[key] = path
    return sorted(
        dedup.values(),
        key=lambda p: (
            p["context"],
            0 if p["states"][-1] == "H2O" else 1,
            len(p["states"]),
            " ".join(p["states"]),
            " ".join(p["node_ids"]),
        ),
    )


def write_merged_manifest(rows: list[dict], out_dir: Path) -> None:
    with open(out_dir / "merged_manifest.csv", "w", encoding="utf-8", newline="") as fh:
        cols = [
            "context",
            "family",
            "label",
            "state",
            "deltaE_family_eV",
            "final_energy_eV",
            "max_force_eVA",
            "site",
            "co_idx",
            "orientation",
            "min_contact_ratio",
            "reaction_neighbors",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def write_graph(nodes: dict[str, dict], edges: list[dict], out_dir: Path) -> None:
    with open(out_dir / "nodes.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["node_id", "context", "family", "label", "state", "site", "co_idx", "final_energy_eV", "max_force_eVA"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for node in nodes.values():
            writer.writerow({col: node.get(col, "") for col in cols})

    with open(out_dir / "edges.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["context", "source", "move", "target", "source_state", "target_state"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(edges)


def write_paths(paths: list[dict], nodes: dict[str, dict], out_dir: Path) -> None:
    rows = []
    text = []
    complete = 0
    partial = 0
    for idx, path in enumerate(paths, start=1):
        path_id = f"path_{idx}"
        path["path_id"] = path_id
        terminal = path["states"][-1]
        status = "complete" if terminal == "H2O" else "partial"
        if status == "complete":
            complete += 1
        else:
            partial += 1
        text.append(f"{path_id} [{path['context']}] ({status}): " + " -> ".join(path["states"]))
        rows.append(
            {
                "path_id": path_id,
                "context": path["context"],
                "status": status,
                "n_steps": len(path["states"]),
                "states": " -> ".join(path["states"]),
                "state_labels": " -> ".join(STATE_LABELS[s] for s in path["states"]),
                "moves": " -> ".join(path["moves"]),
                "start_node": path["node_ids"][0],
                "end_node": path["node_ids"][-1],
                "end_state": terminal,
            }
        )
    with open(out_dir / "all_paths.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path_id", "context", "status", "n_steps", "states", "state_labels", "moves", "start_node", "end_node", "end_state"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "all_paths.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    with open(out_dir / "path_counts.json", "w", encoding="utf-8") as fh:
        json.dump({"complete": complete, "partial": partial, "total": len(paths)}, fh, indent=2, ensure_ascii=False)


def plot_profiles(paths: list[dict], nodes: dict[str, dict], out_dir: Path) -> None:
    profile_dir = out_dir / "energy_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    fig, ax = plt.subplots(figsize=(15, 8))
    max_len = max(len(path["states"]) for path in paths)

    seen_labels = set()
    for idx, path in enumerate(paths, start=1):
        energies = [nodes[nid]["final_energy_eV"] for nid in path["node_ids"]]
        rel = [e - energies[0] for e in energies]
        deltas = [0.0] + [energies[i] - energies[i - 1] for i in range(1, len(energies))]
        x = list(range(len(path["states"])))
        color = CONTEXT_COLORS.get(path["context"], "#444444")
        label = path["context"] if path["context"] not in seen_labels else "_nolegend_"
        seen_labels.add(path["context"])
        linestyle = "-" if path["states"][-1] == "H2O" else "--"
        ax.plot(x, rel, marker="o", linewidth=1.8, linestyle=linestyle, color=color, alpha=0.85, label=label)

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        ax_i.plot(x, rel, marker="o", linewidth=2.0, linestyle=linestyle, color=color)
        ax_i.set_xticks(x, [STATE_LABELS[s] for s in path["states"]])
        ax_i.set_ylabel("Relative Energy to OO* (eV)")
        ax_i.set_title(f"{path['path_id']} [{path['context']}]")
        ax_i.grid(alpha=0.25)
        fig_i.tight_layout()
        fig_i.savefig(profile_dir / f"{path['path_id']}.png", dpi=220)
        plt.close(fig_i)

        with open(profile_dir / f"{path['path_id']}.csv", "w", encoding="utf-8", newline="") as fh:
            cols = [
                "path_id",
                "context",
                "step_index",
                "state",
                "state_label",
                "node_id",
                "final_energy_eV",
                "relative_to_start_eV",
                "delta_from_previous_step_eV",
                "max_force_eVA",
            ]
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for step_index, (state, nid, erel, de) in enumerate(zip(path["states"], path["node_ids"], rel, deltas)):
                rec = {
                    "path_id": path["path_id"],
                    "context": path["context"],
                    "step_index": step_index,
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "node_id": nid,
                    "final_energy_eV": f"{nodes[nid]['final_energy_eV']:.6f}",
                    "relative_to_start_eV": f"{erel:.6f}",
                    "delta_from_previous_step_eV": f"{de:.6f}",
                    "max_force_eVA": f"{nodes[nid]['max_force_eVA']:.6f}",
                }
                writer.writerow(rec)
                combined_rows.append(rec)

    ax.set_xticks(range(max_len), [str(i) for i in range(max_len)])
    ax.set_ylabel("Relative Energy to OO* (eV)")
    ax.set_title("All Merged ORR Structural Energy Profiles")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(profile_dir / "all_paths_profiles.png", dpi=240)
    plt.close(fig)

    with open(profile_dir / "all_paths_profiles.csv", "w", encoding="utf-8", newline="") as fh:
        cols = [
            "path_id",
            "context",
            "step_index",
            "state",
            "state_label",
            "node_id",
            "final_energy_eV",
            "relative_to_start_eV",
            "delta_from_previous_step_eV",
            "max_force_eVA",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(combined_rows)


def write_picture_description(paths: list[dict], nodes: dict[str, dict], out_dir: Path) -> None:
    complete = [p for p in paths if p["states"][-1] == "H2O"]
    partial = [p for p in paths if p["states"][-1] != "H2O"]
    by_context = defaultdict(list)
    for path in paths:
        by_context[path["context"]].append(path)

    lines = []
    lines.append("Picture description")
    lines.append(f"The combined profile overlays {len(paths)} reaction lines from OO* starts across {len(by_context)} surface contexts.")
    lines.append(f"Solid lines terminate at H2O* and dashed lines terminate early at the last reasonable intermediate.")
    lines.append(f"In the merged dataset, {len(complete)} paths are complete and {len(partial)} paths remain partial.")
    lines.append("")
    lines.append("Context summary")
    for context in sorted(by_context):
        cpaths = by_context[context]
        c_complete = sum(1 for p in cpaths if p['states'][-1] == 'H2O')
        c_partial = len(cpaths) - c_complete
        end_states = ", ".join(sorted({STATE_LABELS[p["states"][-1]] for p in cpaths}))
        lines.append(f"- {context}: {len(cpaths)} paths, {c_complete} complete, {c_partial} partial, terminal states = {end_states}.")

    all_drops = []
    for path in paths:
        rel = [nodes[nid]["final_energy_eV"] - nodes[path["node_ids"][0]]["final_energy_eV"] for nid in path["node_ids"]]
        all_drops.append((path["path_id"], path["context"], min(rel), rel[-1], " -> ".join(STATE_LABELS[s] for s in path["states"])))
    all_drops.sort(key=lambda x: x[3])
    lines.append("")
    lines.append("Most stabilized final lines")
    for path_id, context, _min_rel, final_rel, labels in all_drops[:5]:
        lines.append(f"- {path_id} [{context}] ends at {final_rel:.3f} eV relative to its OO* start: {labels}.")

    with open(out_dir / "picture_description.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    nodes, edges, _ = build_graph(rows)
    paths = enumerate_maximal_paths(nodes, edges)
    write_merged_manifest(rows, OUT_DIR)
    write_graph(nodes, edges, OUT_DIR)
    write_paths(paths, nodes, OUT_DIR)
    plot_profiles(paths, nodes, OUT_DIR)
    write_picture_description(paths, nodes, OUT_DIR)
    print(f"Wrote merged network with {len(nodes)} nodes, {len(edges)} edges, and {len(paths)} maximal OO* pathways to {OUT_DIR}")


if __name__ == "__main__":
    main()
