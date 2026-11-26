import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.utils import to_networkx

from Configures import data_args, model_args, train_args
from load_dataset import get_dataset, get_dataloader
from models import GnnNets
from my_mcts import mcts
from utils import PlotUtils

# Choose a completed ProtGNN graph-classification run
DATASET = "MUTAG"       # e.g. "MUTAG", "bbbp", "BA_2Motifs"
MODEL = "GCN"           # "GCN", "GAT", or "GIN"
VARIANT = "ProtGNN"
SEED = 1

if VARIANT != "ProtGNN":
    raise ValueError("This script needs a ProtGNN checkpoint with learned prototypes.")

projection_records = []
data_args.dataset_name = DATASET
data_args.seed = SEED
model_args.model_name = MODEL.lower()
model_args.enable_prot = True

dataset = get_dataset(data_args.dataset_dir, DATASET, task=data_args.task)
model = GnnNets(dataset.num_node_features, int(dataset.num_classes), model_args)

checkpoint_path = (
    PROJECT_ROOT / "Results" / DATASET / VARIANT / MODEL
    / f"seed_{SEED}" / f"{MODEL.lower()}_best.pth"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.update_state_dict(checkpoint["net"])
model.eval()

splits = get_dataloader(
    dataset,
    train_args.batch_size,
    data_split_ratio=data_args.data_split_ratio,
    seed=SEED,
)
train_indices = splits["train"].dataset.indices

class_names = {
    0: "non-mutagenic",
    1: "mutagenic",
}


out_dir = (
    PROJECT_ROOT / "projections" / "results"
    / DATASET / VARIANT / MODEL / f"seed_{SEED}"
)
out_dir.mkdir(parents=True, exist_ok=True)
plotter = PlotUtils(DATASET.lower())

for proto_idx in range(int(dataset.num_classes) * model_args.num_prototypes_per_class):
    class_id = proto_idx // model_args.num_prototypes_per_class
    class_name = class_names.get(class_id, f"class_{class_id}")
    best = None
    matching_graphs = 0

    for graph_idx in train_indices[proto_idx * 10:]:
        graph_data = dataset[graph_idx]
        if int(graph_data.y.item()) != class_id:
            continue

        coalition, similarity, _ = mcts(
            graph_data, model, model.model.prototype_vectors[proto_idx]
        )
        matching_graphs += 1

        if best is None or similarity > best[0]:
            best = (similarity, graph_idx, graph_data, coalition)

        if matching_graphs == 10:
            break

    if best is None:
        print(f"Skipped prototype {proto_idx}: no matching training graph found.")
        continue

    similarity, selected_graph_idx, graph_data, coalition = best
    graph = to_networkx(graph_data, to_undirected=True)
    image_path = out_dir / f"prototype_{proto_idx}_{class_name}.png"

    plotter.plot(graph, coalition, figname=str(image_path), x=graph_data.x)

    projection_records.append({
        "prototype_index": proto_idx,
        "class_id": class_id,
        "class_name": class_name,
        "graph_index": int(selected_graph_idx),
        "similarity": float(similarity),
        "selected_atom_indices": [int(i) for i in coalition],
        "image": str(image_path),
    })
    print(f"Saved {image_path} — class: {class_name}, similarity={similarity:.4f}")

manifest_path = out_dir / "projection_manifest.json"
manifest_path.write_text(json.dumps(projection_records, indent=2))
print(f"Saved projection details to {manifest_path}")