"""Render HIV ProtGNN prototype subgraphs as highlighted molecules."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from rdkit import Chem
from rdkit.Chem import Draw

from Configures import data_args, model_args, train_args
from load_dataset import get_dataset, get_dataloader
from models import GnnNets
from my_mcts import mcts


# Match these to a completed HIV ProtGNN run.
DATASET = "HIV"
MODEL = "GCN"  # "GCN", "GAT", or "GIN"
VARIANT = "ProtGNN"
SEED = 1
MAX_GRAPHS_PER_PROTOTYPE = 10

if VARIANT != "ProtGNN":
    raise ValueError("This script requires a ProtGNN checkpoint with learned prototypes.")

data_args.dataset_name = DATASET
data_args.seed = SEED
model_args.model_name = MODEL.lower()
model_args.enable_prot = True

dataset = get_dataset(data_args.dataset_dir, DATASET, task=data_args.task)
model = GnnNets(dataset.num_node_features, int(dataset.num_classes), model_args)
checkpoint_path = (
    PROJECT_ROOT / "Results" / DATASET / VARIANT / MODEL
    / f"seed_{SEED}" / f"{MODEL.lower()}_latest.pth"
)
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"ProtGNN checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.update_state_dict(checkpoint["net"])
model.eval()

splits = get_dataloader(
    dataset,
    train_args.batch_size,
    data_split_ratio=data_args.data_split_ratio,
    seed=SEED,
)
train_indices = list(splits["train"].dataset.indices)

class_names = {0: "inactive", 1: "active"}
out_dir = (
    PROJECT_ROOT / "projections" / "results"
    / DATASET / VARIANT / MODEL / f"seed_{SEED}"
)
out_dir.mkdir(parents=True, exist_ok=True)
projection_records = []

# These indices follow PyG MoleculeNet's atom/bond feature encoding.
BOND_TYPES = [
    Chem.BondType.UNSPECIFIED, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE, Chem.BondType.QUADRUPLE, Chem.BondType.QUINTUPLE,
    Chem.BondType.HEXTUPLE, Chem.BondType.ONEANDAHALF,
    Chem.BondType.TWOANDAHALF, Chem.BondType.THREEANDAHALF,
    Chem.BondType.FOURANDAHALF, Chem.BondType.FIVEANDAHALF,
    Chem.BondType.AROMATIC, Chem.BondType.IONIC, Chem.BondType.HYDROGEN,
    Chem.BondType.THREECENTER, Chem.BondType.DATIVEONE, Chem.BondType.DATIVE,
    Chem.BondType.DATIVEL, Chem.BondType.DATIVER, Chem.BondType.OTHER,
    Chem.BondType.ZERO,
]


def graph_to_molecule(graph_data):
    """Convert PyG HIV atom/bond features into an RDKit molecule for drawing."""
    editable = Chem.RWMol()
    for atom_features in graph_data.x:
        atomic_number = int(atom_features[0].item())
        atom = Chem.Atom(atomic_number)
        editable.AddAtom(atom)

    edge_attrs = getattr(graph_data, "edge_attr", None)
    seen = set()
    for edge_pos in range(graph_data.edge_index.shape[1]):
        source = int(graph_data.edge_index[0, edge_pos].item())
        target = int(graph_data.edge_index[1, edge_pos].item())
        pair = (min(source, target), max(source, target))
        if source == target or pair in seen:
            continue
        seen.add(pair)
        bond_type_index = int(edge_attrs[edge_pos, 0].item()) if edge_attrs is not None else 1
        if bond_type_index >= len(BOND_TYPES):
            bond_type_index = 1
        editable.AddBond(source, target, BOND_TYPES[bond_type_index])

    molecule = editable.GetMol()
    # Sanitization may fail for unusual feature combinations; drawing can still work.
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        pass
    return molecule


for prototype_index in range(int(dataset.num_classes) * model_args.num_prototypes_per_class):
    class_id = prototype_index // model_args.num_prototypes_per_class
    class_name = class_names.get(class_id, f"class_{class_id}")
    best = None
    matching_graphs = 0

    # Search up to the configured number of training molecules of this class.
    for graph_index in train_indices:
        graph_data = dataset[graph_index]
        if int(graph_data.y.reshape(-1)[0].item()) != class_id:
            continue

        coalition, similarity, _ = mcts(
            graph_data, model, model.model.prototype_vectors[prototype_index]
        )
        matching_graphs += 1
        if best is None or similarity > best[0]:
            best = (similarity, graph_index, graph_data, coalition)
        if matching_graphs >= MAX_GRAPHS_PER_PROTOTYPE:
            break

    if best is None:
        print(f"Skipped prototype {prototype_index}: no matching training molecule found.")
        continue

    similarity, graph_index, graph_data, coalition = best
    molecule = graph_to_molecule(graph_data)
    selected_atoms = sorted({int(atom_index) for atom_index in coalition})
    selected_bonds = [
        bond.GetIdx() for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in selected_atoms
        and bond.GetEndAtomIdx() in selected_atoms
    ]
    image_path = out_dir / f"prototype_{prototype_index}_{class_name}.png"
    Draw.MolToFile(
        molecule,
        str(image_path),
        size=(900, 700),
        highlightAtoms=selected_atoms,
        highlightBonds=selected_bonds,
    )

    projection_records.append({
        "prototype_index": prototype_index,
        "class_id": class_id,
        "class_name": class_name,
        "graph_index": int(graph_index),
        "similarity": float(similarity),
        "selected_atom_indices": selected_atoms,
        "image": str(image_path),
    })
    print(
        f"Saved {image_path} — class: {class_name}, graph: {graph_index}, "
        f"similarity={similarity:.4f}"
    )

manifest_path = out_dir / "projection_manifest.json"
manifest_path.write_text(json.dumps(projection_records, indent=2), encoding="utf-8")
print(f"Saved projection details to {manifest_path}")
