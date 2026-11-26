"""Score the last saved ProtGNN checkpoints with the same test graphs and cues."""

import argparse
import csv
from pathlib import Path

import torch

from Configures import model_args, train_args
from models import GnnNets
from .atlas import add_cue, load_graphs
from .audit import audit_predictions, score_predictions
from .run import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    root = parser.parse_args().results
    with (root / "splits.csv").open(encoding="utf-8") as stream:
        test_ids = {}
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                test_ids.setdefault(int(row["seed"]), []).append(int(row["atlas_id"]))

    torch.set_num_threads(1)
    model_args.model_name = "gcn"
    model_args.device = torch.device("cpu")
    model_args.enable_prot = True
    clean, _ = load_graphs()
    all_scores = []
    all_predictions = []
    for seed, ids in test_ids.items():
        for variant, graphs in (("ProtGNN", clean), ("ProtGNN-Cue", add_cue(clean, seed))):
            folder = root / variant / "GCN" / f"seed_{seed}"
            checkpoint = torch.load(folder / "gcn_latest.pth", map_location="cpu", weights_only=False)
            epoch = checkpoint["epoch"]
            model = GnnNets(graphs.num_node_features, graphs.num_classes, model_args)
            # All ten vectors must have been projected by the saved epoch.
            log = (folder / "train.log").read_text(encoding="utf-8")
            through_epoch, marker, _ = log.partition(f"Train Epoch:{epoch} ")
            completed = through_epoch.count("Projection of prototype completed")
            if not marker or epoch < train_args.proj_epochs or completed < model.model.num_prototypes:
                raise ValueError(f"{variant} seed {seed} has no complete projection by epoch {epoch}")
            model.load_state_dict(checkpoint["net"])
            model.eval()
            by_id = {g.atlas_id: g for g in graphs}
            rows = audit_predictions(model, [by_id[i] for i in ids])
            run = {"variant": variant, "seed": seed, "epoch": epoch}
            all_scores.append({**run, **score_predictions(rows)})
            all_predictions.extend({**run, **row} for row in rows)
    write_csv(root / "after-projection.csv", all_scores)
    write_csv(root / "after-projection-predictions.csv", all_predictions)


if __name__ == "__main__":
    main()
