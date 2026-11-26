"""Run from the repository root: python -m extensions.atlas_bridges.run --output atlas-rerun"""

import argparse
import csv
import json
import platform
import random
import time
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean, stdev
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch
import torch_geometric

from Configures import data_args, model_args, train_args
from load_dataset import get_dataloader
from models import GnnNets, train_gnns
from .atlas import add_cue, load_graphs
from .audit import audit_predictions, retrieve_prototypes, score_predictions


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_scores(root, runs):
    all_seeds = []
    for variant, records in runs.items():
        folder = root / variant / "GCN"
        write_csv(folder / "summary.csv", records)
        for row in records:
            all_seeds.append({"variant": variant, **row, "nearest_proto_flip": row.get("nearest_proto_flip")})

        summary = {
            "dataset": "Atlas-Bridges", "variant": variant, "model": "GCN",
            "num_seeds": len(records), "seeds": [r["seed"] for r in records], "runs": records,
        }
        for metric in records[0]:
            if metric in ("seed", "log_file"):
                continue
            values = [row[metric] for row in records]
            key = "acc" if metric == "test_acc" else metric
            summary["mean_" + key] = mean(values)
            summary["std_" + key] = stdev(values) if len(values) > 1 else None
        write_json(folder / "summary.json", summary)
    # The group's result collector can rewrite the individual model summaries.
    write_csv(root / "seed-metrics.csv", all_seeds)


def run_one(folder, graphs, split, variant, seed):
    folder.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_args.seed = seed
    model_args.enable_prot = variant.startswith("ProtGNN")
    train_args.save_dir = str(folder)

    # Give the existing trainer our dataset and keep its shared hyper_search log private.
    start = time.perf_counter()
    with (folder / "train.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        print(f"Atlas-Bridges | {variant} | GCN | seed {seed}")
        with (patch.object(train_gnns, "get_dataset", return_value=graphs),
              patch.object(train_gnns, "append_record", lambda message: None)):
            result = train_gnns.train_GC(clst=0.10, sep=0.05)
    runtime = time.perf_counter() - start

    # This checkpoint was written above by train_GC and includes a NumPy validation score.
    checkpoint = torch.load(folder / "gcn_best.pth", map_location="cpu", weights_only=False)
    model = GnnNets(graphs.num_node_features, graphs.num_classes, model_args)
    model.load_state_dict(checkpoint["net"])
    model.eval()
    rows = audit_predictions(model, [graphs[i] for i in split["test"]])
    write_csv(folder / "predictions.csv", rows)
    if model_args.enable_prot:
        examples = retrieve_prototypes(model, graphs, split["train"], model_args.num_prototypes_per_class)
        write_json(folder / "prototypes.json", examples)

    record = {
        "seed": seed,
        "test_acc": float(result["test_acc"]),
        "test_loss": float(result["test_loss"]),
        "runtime_seconds": runtime,
        "log_file": f"{variant}/GCN/seed_{seed}/train.log",
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": float(checkpoint["acc"]),
    }
    record.update(score_predictions(rows))
    assert abs(record["observed_acc"] - record["test_acc"]) < 1e-6
    majority = int(mean(graphs[i].y.item() for i in split["train"]) > 0.5)
    record["majority_acc"] = mean(r["label"] == majority for r in rows)
    record["leaf_rule_acc"] = mean(r["label"] == r["has_leaf"] for r in rows)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory for this run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each seed must be unique.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data_args.dataset_name = "Atlas-Bridges"
    model_args.model_name = "gcn"
    model_args.device = torch.device("cpu")
    train_args.max_epochs = 500
    clean, catalog = load_graphs()
    write_csv(args.output / "graphs.csv", catalog)
    write_json(args.output / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "pyg": torch_geometric.__version__,
        "numpy": np.__version__, "networkx": nx.__version__, "platform": platform.platform(),
        "device": "cpu", "torch_threads": torch.get_num_threads(), "seeds": args.seeds,
    })

    runs = {name: [] for name in ("Original", "ProtGNN", "Original-Cue", "ProtGNN-Cue")}
    split_rows = []
    for seed in args.seeds:
        marked = add_cue(clean, seed)
        loaders = get_dataloader(clean, train_args.batch_size, data_split_ratio=data_args.data_split_ratio, seed=seed)
        split = {name: list(loader.dataset.indices) for name, loader in loaders.items()}
        for part, indices in split.items():
            for i in indices:
                split_rows.append({"seed": seed, "split": part, "atlas_id": clean[i].atlas_id,
                                   "cue": int(marked[i].x[0, 2])})
        for variant, records in runs.items():
            graphs = marked if variant.endswith("-Cue") else clean
            folder = args.output / variant / "GCN" / f"seed_{seed}"
            print(f"Training {variant}, seed {seed}", flush=True)
            records.append(run_one(folder, graphs, split, variant, seed))
            print(f"Test accuracy {records[-1]['test_acc']:.4f}", flush=True)

    write_csv(args.output / "splits.csv", split_rows)
    save_scores(args.output, runs)


if __name__ == "__main__":
    main()
