import time # efficiency study
import csv
import json
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

# Resolve project root so imports work even if the script lives inside Results/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Results" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Configures import data_args, model_args, train_args
from models.train_gnns import train_GC

DATASET = "Graph-SST2" #MUTAG, bbbp, Graph-SST2, BA_2Motifs
MODEL = "GAT" #GCN, GAT, GIN
SEEDS = [1, 2, 3] #random seeds

# Run one by one by keeping a single item in RUN_STEPS, or run all three in order:
# ["Original", "ProtGNN", "ProtGNN+"]
RUN_STEPS = ["Original"]


def configure_variant(variant: str) -> None:
    # Reset sampling flags first so state never leaks from one run to the next
    for attr in ("enable_prot_plus", "prot_plus", "subgraph_sampling"):
        if hasattr(model_args, attr):
            setattr(model_args, attr, False)
        if hasattr(train_args, attr):
            setattr(train_args, attr, False)

    if variant == "Original":
        model_args.enable_prot = False
        train_args.warm_epochs = 10
    elif variant == "ProtGNN":
        model_args.enable_prot = True
        train_args.warm_epochs = 10
    elif variant == "ProtGNN+":
        model_args.enable_prot = True
        train_args.warm_epochs = 200
        for attr in ("enable_prot_plus", "prot_plus", "subgraph_sampling"):
            if hasattr(model_args, attr):
                setattr(model_args, attr, True)
            if hasattr(train_args, attr):
                setattr(train_args, attr, True)
    else:
        raise ValueError(f"Unknown variant: {variant}")


results_root = PROJECT_ROOT / "Results"
data_args.dataset_name = DATASET
model_args.model_name = MODEL.lower()
train_args.max_epochs = 500

for variant in RUN_STEPS:
    out_dir = results_root / DATASET / variant / MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_variant(variant)

    records = []

    for seed in SEEDS:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_path = seed_dir / "train.log"

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        data_args.seed = seed

        
        train_args.save_dir = str(seed_dir)
        start_time = time.perf_counter() #efficiency study 
        
        with open(log_path, "w", encoding="utf-8") as f_log, \
             redirect_stdout(f_log), \
             redirect_stderr(f_log):
            print(f"=== Dataset={DATASET} | Variant={variant} | Model={MODEL} | Seed={seed} ===")
            result = train_GC(clst=0.10, sep=0.05)
        
        runtime_seconds = time.perf_counter() - start_time # efficiency study. records elapsed seconds
        
        test_acc = None
        if isinstance(result, dict):
            for key in ("test_acc", "acc_test", "best_test_acc"):
                if key in result:
                    test_acc = float(result[key])
                    break
        elif isinstance(result, (tuple, list)) and len(result) > 0:
            try:
                test_acc = float(result[-1])
            except (TypeError, ValueError):
                pass

        records.append({
            "seed": seed,
            "test_acc": test_acc,
            "runtime_seconds": runtime_seconds,
            "log_file": str(log_path),
        })
        print(f"[{variant} | Seed {seed}] Done -> {log_path}")

    valid_accs = [r["test_acc"] for r in records if r["test_acc"] is not None]
    summary = {
        "dataset": DATASET,
        "variant": variant,
        "model": MODEL,
        "seeds": SEEDS,
        "runs": records,
        "mean_acc": float(np.mean(valid_accs)) if valid_accs else None,
        "std_acc": float(np.std(valid_accs, ddof=1)) if len(valid_accs) > 1 else (0.0 if valid_accs else None),
        "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in records])), ##efficiency studies : runtime_seconds
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f_json:
        json.dump(summary, f_json, indent=2)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["seed", "test_acc", "runtime_seconds", "log_file"]) #efficiency studies : runtime_seconds
        writer.writeheader()
        writer.writerows(records)