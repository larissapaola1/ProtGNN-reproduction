"""Check the retrieved training examples and draw a held-out prediction."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def write_coverage(root, graphs):
    # A bridge is retained only when the mask keeps both of its original endpoints.
    with (root / "prototype-coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["variant", "seed", "prototype", "atlas_id", "size",
                         "retains_bridge", "random_connected_hit_rate"])
        for path in sorted(root.glob("ProtGNN*/GCN/seed_*/prototypes.json")):
            for proto in json.loads(path.read_text(encoding="utf-8")):
                if proto["class"] == 0:
                    continue
                graph = graphs[str(proto["atlas_id"])]
                bridges = [set(edge) for edge in nx.bridges(graph)]
                nodes = set(proto["nodes"])
                hits = []
                for mask in combinations(graph, len(nodes)):
                    if nx.is_connected(graph.subgraph(mask)):
                        hits.append(any(edge.issubset(mask) for edge in bridges))
                writer.writerow([path.parents[2].name, path.parent.name.removeprefix("seed_"),
                                 proto["prototype"], proto["atlas_id"], len(nodes),
                                 int(any(edge <= nodes for edge in bridges)), sum(hits) / len(hits)])


def draw_graph(ax, graph, selected):
    layout = nx.spring_layout(graph, seed=4, iterations=100)
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    colors = ["#bc5727" if frozenset(edge) in bridges else "#687486" for edge in graph.edges]
    nodes = ["#82c7bc" if node in selected else "#e4e9ed" for node in graph]
    nx.draw_networkx(graph, pos=layout, ax=ax, node_color=nodes, edge_color=colors,
                     width=2, node_size=430, font_size=9, edgecolors="white")
    ax.margins(0.22)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    folder = args.results / "ProtGNN-Cue" / "GCN" / f"seed_{args.seed}"
    with (args.results / "graphs.csv").open(encoding="utf-8") as stream:
        graphs = {row["atlas_id"]: nx.from_graph6_bytes(row["graph6"].encode())
                  for row in csv.DictReader(stream)}
    write_coverage(args.results, graphs)
    with (folder / "predictions.csv").open(encoding="utf-8") as stream:
        changed = [r for r in csv.DictReader(stream)
                   if r["label"] == "1" and r["aligned_pred"] != r["reversed_pred"]]
    if not changed:
        print("No bridge-positive prediction flipped in this seed.")
        return
    row = min(changed, key=lambda r: int(r["atlas_id"]))
    prototypes = json.loads((folder / "prototypes.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    graph = graphs[row["atlas_id"]]
    draw_graph(axes[0], graph, [])
    axes[0].set_title(f"Test graph {row['atlas_id']}\nContains a bridge", fontsize=11)
    for ax, view in zip(axes[1:], ("aligned", "reversed")):
        proto_id = int(row[view + "_nearest_proto"])
        proto = prototypes[proto_id]
        draw_graph(ax, graphs[str(proto["atlas_id"])], proto["nodes"])
        decision = "bridge" if row[view + "_pred"] == "1" else "no bridge"
        ax.set_title(f"{view.capitalize()} marker predicts {decision}\n"
                     f"Closest prototype {proto_id}, training graph {proto['atlas_id']}", fontsize=10)
    fig.suptitle("Changing one marker on the same graph", fontsize=16, fontweight="bold")
    legend = [Line2D([], [], color="#bc5727", lw=2, label="Bridge in source graph"),
              Line2D([], [], marker="o", linestyle="", color="#82c7bc", markersize=9,
                     label="Nodes selected for training exemplar")]
    fig.legend(handles=legend, loc="lower center", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(folder / "prototype-example.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
