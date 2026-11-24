import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Results" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Configures import data_args, model_args, train_args
# Adjust these imports to match your repo's dataset & model constructors:
# from load_dataset import get_dataloader, get_dataset
# from models.GnnNets import GnnNets


def extract_graph_embedding(model, batch, out):
    """Return graph-level embedding tensor `h` with shape (batch_size, embed_dim)."""
    # Common patterns in ProtGNN / GnnNets codebases—uncomment or adapt to your model:
    # 1. If your model has a dedicated method/attribute:
    #    return model.get_graph_embedding(batch)
    # 2. If `out` is a tuple/dict containing embeddings:
    #    return out[1]  # e.g. (logits, graph_emb, ...)
    if hasattr(model, "last_graph_emb") and model.last_graph_emb is not None:
        return model.last_graph_emb
    if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
        return out[1]
    raise RuntimeError(
        "Could not automatically find graph embeddings `h`. "
        "Inspect `model` / `out` and update `extract_graph_embedding()`."
    )


def extract_prototypes(model, num_classes):
    """Return (prot_vectors, prot_labels) arrays for learned prototypes."""
    # Common attribute names across ProtGNN implementations:
    candidates = [
        "prototype_vectors",
        "prototypes",
    ]
    prot_tensor = None
    for attr in candidates:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if torch.is_tensor(val):
                prot_tensor = val
                break
        if hasattr(model, "prot_layer") and hasattr(model.prot_layer, attr):
            val = getattr(model.prot_layer, attr)
            if torch.is_tensor(val):
                prot_tensor = val
                break

    if prot_tensor is None:
        raise RuntimeError(
            "Could not automatically find prototype tensor. "
            "Check `model` / `model.prot_layer` attributes and update `extract_prototypes()`."
        )

    prot_np = prot_tensor.detach().cpu().numpy()
    # If prototypes are ordered evenly across classes (e.g., k prototypes per class):
    n_prot = prot_np.shape[0]
    if n_prot % num_classes == 0:
        per_class = n_prot // num_classes
        prot_labels = np.repeat(np.arange(num_classes), per_class)
    else:
        prot_labels = np.zeros(n_prot, dtype=int)
    return prot_np, prot_labels


def build_mutag_loader():
    """Construct a DataLoader over MUTAG graphs."""
    # Example patterns—uncomment the one matching your `load_dataset.py`:
    # dataset = get_dataset(data_args)
    # return DataLoader(dataset, batch_size=32, shuffle=False)
    #
    # train_loader, val_loader, test_loader = get_dataloader(data_args)
    # return train_loader
    raise NotImplementedError(
        "Wire `build_mutag_loader()` to your `load_dataset.py` dataloader/dataset."
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the checkpoint configuration
    data_args.dataset_name = "MUTAG"
    model_args.model_name = "gcn"
    model_args.enable_prot = True

    loader = build_mutag_loader()

    # Instantiate GnnNets and load saved checkpoint
    # model = GnnNets(...).to(device)
    ckpt_path = PROJECT_ROOT / "checkpoint" / data_args.dataset_name / f"{model_args.model_name}_best.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    graph_embeds = []
    graph_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            h = extract_graph_embedding(model, batch, out)
            graph_embeds.append(h.detach().cpu().numpy())
            graph_labels.append(batch.y.view(-1).detach().cpu().numpy())

    X_graphs = np.concatenate(graph_embeds, axis=0)
    y_graphs = np.concatenate(graph_labels, axis=0)
    num_classes = int(len(np.unique(y_graphs)))

    X_protos, y_protos = extract_prototypes(model, num_classes=num_classes)

    # Project graph embeddings and prototypes together into 2D
    X_all = np.vstack([X_graphs, X_protos])
    # MUTAG is small, so keep perplexity well below total points (default 30 can be too high)
    perplexity = min(15, max(2, len(X_all) // 4))

    coords_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_all)

    n_g = len(X_graphs)
    g_2d = coords_2d[:n_g]
    p_2d = coords_2d[n_g:]

    # Use a consistent color palette per class for both graph dots and prototype stars
    cmap = plt.get_cmap("tab10")
    classes = np.unique(y_graphs)

    plt.figure(figsize=(7, 6))
    for idx, cls in enumerate(classes):
        color = cmap(idx % 10)
        g_mask = y_graphs == cls
        plt.scatter(
            g_2d[g_mask, 0],
            g_2d[g_mask, 1],
            color=color,
            marker="o",
            s=28,
            alpha=0.75,
            label=f"Class {cls} (graphs)",
        )

        p_mask = y_protos == cls
        if np.any(p_mask):
            plt.scatter(
                p_2d[p_mask, 0],
                p_2d[p_mask, 1],
                color=color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                s=240,
                label=f"Class {cls} (prototypes)",
            )

    plt.title("t-SNE on MUTAG: Graph Embeddings (Dots) & Prototypes (Stars)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_path = SCRIPT_DIR / "mutag_protgnn_tsne.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved t-SNE figure to {out_path}")


if __name__ == "__main__":
    main()