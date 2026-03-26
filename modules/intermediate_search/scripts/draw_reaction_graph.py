import csv
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = ROOT / "intermediates" / "orr_search_chgnet_pbe" / "reaction_network"
OUT_DIR = NETWORK_DIR / "graph_viz"


STATE_ORDER = ["O2", "O", "OH", "OOH", "H2O2", "H2O"]
STATE_LABELS = {
    "O2": "OO*",
    "O": "O*",
    "OH": "OH*",
    "OOH": "OOH*",
    "H2O2": "H2O2*",
    "H2O": "H2O*",
}
STATE_COLORS = {
    "O2": "#1d3557",
    "O": "#457b9d",
    "OH": "#2a9d8f",
    "OOH": "#e9c46a",
    "H2O2": "#f4a261",
    "H2O": "#e76f51",
}
CONTEXT_ORDER = [
    "direct_co_top",
    "direct_hollow",
    "direct_co_o_bridge",
    "vacancy_edge_bridge",
    "vacancy_center",
    "vacancy_co_top_3",
    "vacancy_co_top_13",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def prettify_context(context: str) -> str:
    return (
        context.replace("direct_", "direct\n")
        .replace("vacancy_", "vacancy\n")
        .replace("_co_top_", "\nco_top_")
        .replace("_co_o_bridge", "\nco-o bridge")
        .replace("_edge_bridge", "\nedge bridge")
        .replace("_center", "\ncenter")
        .replace("_hollow", "\nhollow")
        .replace("_co_top", "\nco-top")
    )


def build_graph(nodes_rows: list[dict], edges_rows: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in nodes_rows:
        g.add_node(
            row["node_id"],
            state=row["state"],
            context=row["family"].rsplit("__", 1)[0],
            energy=float(row["final_energy_eV"]),
            fmax=float(row["max_force_eVA"]),
            site=row["site"],
            label=row["label"],
        )
    for row in edges_rows:
        if row["source"] in g and row["target"] in g:
            g.add_edge(row["source"], row["target"], move=row["move"], context=row["context"])
    return g


def layout_positions(g: nx.DiGraph) -> dict[str, tuple[float, float]]:
    contexts_present = [ctx for ctx in CONTEXT_ORDER if any(data["context"] == ctx for _, data in g.nodes(data=True))]
    extra_contexts = sorted({data["context"] for _, data in g.nodes(data=True)} - set(contexts_present))
    contexts_present.extend(extra_contexts)

    pos = {}
    for x_idx, context in enumerate(contexts_present):
        nodes = [(nid, data) for nid, data in g.nodes(data=True) if data["context"] == context]
        states_here = [s for s in STATE_ORDER if any(data["state"] == s for _, data in nodes)]
        for y_idx, state in enumerate(states_here):
            same_state = [nid for nid, data in nodes if data["state"] == state]
            for spread_idx, nid in enumerate(same_state):
                pos[nid] = (x_idx * 3.4 + spread_idx * 0.35, -y_idx * 1.5)
    return pos


def draw_graph(g: nx.DiGraph, out_png: Path, out_svg: Path) -> None:
    pos = layout_positions(g)
    fig, ax = plt.subplots(figsize=(18, 10))

    # Draw context separators and labels.
    contexts = []
    for nid, (x, _y) in pos.items():
        ctx = g.nodes[nid]["context"]
        contexts.append((ctx, x))
    seen = {}
    for ctx, x in contexts:
        seen.setdefault(ctx, []).append(x)
    for ctx, xs in seen.items():
        x_mid = sum(xs) / len(xs)
        ax.text(x_mid, 1.2, prettify_context(ctx), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axvline(min(xs) - 0.55, color="#dddddd", linewidth=1, zorder=0)
    ax.axvline(max(x for x, _ in pos.values()) + 0.85, color="#dddddd", linewidth=1, zorder=0)

    # State row labels.
    states_present = [s for s in STATE_ORDER if any(data["state"] == s for _, data in g.nodes(data=True))]
    for y_idx, state in enumerate(states_present):
        ax.text(-1.2, -y_idx * 1.5, STATE_LABELS[state], ha="right", va="center", fontsize=11, color="#333333")
        ax.axhline(-y_idx * 1.5, color="#f1f1f1", linewidth=1, zorder=0)

    edge_colors = {" +H": "#d62828", "-H": "#f77f00", "+O": "#2a9d8f", "-O": "#577590"}
    default_edge_color = "#888888"
    edge_labels = {}
    for u, v, data in g.edges(data=True):
        move = data["move"]
        edge_labels[(u, v)] = move
        color = edge_colors.get(move, default_edge_color)
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(u, v)],
            edge_color=color,
            width=1.8,
            arrows=True,
            arrowsize=14,
            alpha=0.85,
            ax=ax,
            connectionstyle="arc3,rad=0.08",
        )

    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, rotate=False, ax=ax, label_pos=0.55)

    for state in states_present:
        nodelist = [nid for nid, data in g.nodes(data=True) if data["state"] == state]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodelist,
            node_color=STATE_COLORS[state],
            node_size=1300,
            edgecolors="#222222",
            linewidths=1.0,
            ax=ax,
        )

    node_text = {}
    for nid, data in g.nodes(data=True):
        force_tag = "" if data["fmax"] < 0.011 else "\nF~"
        node_text[nid] = f"{STATE_LABELS[data['state']]}\n{data['site']}{force_tag}"
    nx.draw_networkx_labels(g, pos, labels=node_text, font_size=8, font_color="white", ax=ax)

    ax.set_title("Reaction Graph from Reasonable ORR Structures", fontsize=15)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def write_graph_summary(g: nx.DiGraph, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"nodes: {g.number_of_nodes()}\n")
        fh.write(f"edges: {g.number_of_edges()}\n")
        fh.write("contexts:\n")
        contexts = sorted({data["context"] for _, data in g.nodes(data=True)})
        for ctx in contexts:
            sub = [nid for nid, data in g.nodes(data=True) if data["context"] == ctx]
            states = sorted({g.nodes[nid]["state"] for nid in sub}, key=lambda s: STATE_ORDER.index(s))
            fh.write(f"  - {ctx}: {', '.join(states)}\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_rows = read_csv(NETWORK_DIR / "nodes.csv")
    edges_rows = read_csv(NETWORK_DIR / "edges.csv")
    g = build_graph(nodes_rows, edges_rows)
    draw_graph(g, OUT_DIR / "reaction_graph.png", OUT_DIR / "reaction_graph.svg")
    write_graph_summary(g, OUT_DIR / "graph_summary.txt")
    print(f"Wrote reaction graph visualization to {OUT_DIR}")


if __name__ == "__main__":
    main()
