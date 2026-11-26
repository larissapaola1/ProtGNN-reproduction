"""Evaluate the saved checkpoint on the same test graphs with different markers."""

from statistics import mean

import networkx as nx
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx

from my_mcts import mcts


@torch.no_grad()
def audit_predictions(model, held_out):
    batch = Batch.from_data_list(held_out)
    labels = batch.y
    sign = 2 * labels - 1
    observed = batch.x[batch.ptr[:-1], 2].clone()
    rows = []
    for graph, cue in zip(held_out, observed):
        rows.append({
            "atlas_id": graph.atlas_id,
            "label": graph.y.item(),
            "has_leaf": int((graph.x[:, 1] == 1 / 6).any()),
            "observed_cue": int(cue),
        })

    for view, cue in {"observed": observed, "masked": torch.zeros_like(sign),
                      "aligned": sign, "reversed": -sign}.items():
        batch.x[:, 2] = cue[batch.batch].float()
        logits, probs, _, _, distances = model(batch)
        if not torch.isfinite(probs).all():
            raise ValueError(f"Non-finite probability in the {view} view")
        predictions = logits.argmax(dim=1)
        for i, row in enumerate(rows):
            row[view + "_pred"] = predictions[i].item()
            row[view + "_p_bridge"] = probs[i, 1].item()
            if model.model.enable_prot:
                row[view + "_nearest_proto"] = distances[i].argmin().item()
    return rows


def score_predictions(rows):
    scores = {}
    for view in ("observed", "masked", "aligned", "reversed"):
        scores[view + "_acc"] = mean(row[view + "_pred"] == row["label"] for row in rows)
    recalls = []
    for label in (0, 1):
        recalls.append(mean(r["observed_pred"] == label for r in rows if r["label"] == label))
    scores["balanced_acc"] = mean(recalls)
    scores["prediction_flip"] = mean(r["aligned_pred"] != r["reversed_pred"] for r in rows)
    if "aligned_nearest_proto" in rows[0]:
        scores["nearest_proto_flip"] = mean(
            r["aligned_nearest_proto"] != r["reversed_nearest_proto"] for r in rows
        )
    return scores


@torch.no_grad()
def retrieve_prototypes(model, graphs, train_ids, per_class):
    """Repeat the group's MCTS exemplar search at the selected checkpoint."""
    rows = []
    for proto_id, vector in enumerate(model.model.prototype_vectors):
        label = proto_id // per_class
        candidates = [i for i in train_ids[10 * proto_id:] if graphs[i].y.item() == label]
        best = None
        for i in candidates[:10]:
            graph = graphs[i]
            nodes, score, _ = mcts(graph, model, vector)
            if best is None or score > best["similarity"]:
                source = to_networkx(graph, to_undirected=True)
                best = {
                    "prototype": proto_id, "class": label, "atlas_id": graph.atlas_id,
                    "nodes": nodes, "similarity": score, "cue": int(graph.x[0, 2]),
                    "source_bridges": sorted([sorted(edge) for edge in nx.bridges(source)]),
                }
        rows.append(best)
    return rows
