import csv
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "Results"
    else Path(__file__).resolve().parent
)
RESULTS_ROOT = PROJECT_ROOT / "Results"

TEST_PATTERN = re.compile(
    r"Test:\s*\|\s*Loss:\s*([0-9.eE+-]+)\s*\|\s*Acc:\s*([0-9.eE+-]+)"
)


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = TEST_PATTERN.findall(text)
    if not matches:
        return None, None
    loss_str, acc_str = matches[-1]
    return float(loss_str), float(acc_str)


rows_by_group = {}
all_seeds = set()
runtime_by_run = {}

# Read runtimes written by run_experiments.py before this script rewrites summary.csv.
for summary_path in sorted(RESULTS_ROOT.rglob("summary.csv")):
    rel = summary_path.relative_to(RESULTS_ROOT)
    parts = rel.parts
    if len(parts) < 4:
        continue

    dataset, variant, model = parts[-4:-1]
    with open(summary_path, newline="", encoding="utf-8") as f_csv:
        for row in csv.DictReader(f_csv):
            runtime = row.get("runtime_seconds")
            try:
                seed = int(row["seed"])
                runtime_by_run[(dataset, variant, model, seed)] = float(runtime)
            except (KeyError, TypeError, ValueError):
                continue

# Expected layout: Results/<dataset>/<variant>/<model>/seed_<n>/train.log
for log_path in sorted(RESULTS_ROOT.rglob("train.log")):
    rel = log_path.relative_to(RESULTS_ROOT)
    parts = rel.parts
    if len(parts) < 5 or not parts[-2].startswith("seed_"):
        continue

    dataset, variant, model, seed_folder, _ = parts[-5:]
    try:
        seed = int(seed_folder.replace("seed_", ""))
    except ValueError:
        continue

    all_seeds.add(seed)
    test_loss, test_acc = parse_log(log_path)
    group_key = (dataset, variant, model)
    rows_by_group.setdefault(group_key, []).append({
        "seed": seed,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "runtime_seconds": runtime_by_run.get((dataset, variant, model, seed)),
        "log_file": str(log_path),
    })

ordered_seeds = sorted(all_seeds)
table_rows = []
efficiency_rows = []

for (dataset, variant, model), runs in sorted(rows_by_group.items()):
    runs.sort(key=lambda r: r["seed"])
    accs = [r["test_acc"] for r in runs if r["test_acc"] is not None]

    mean_acc = float(np.mean(accs)) if accs else None
    std_acc = (
        float(np.std(accs, ddof=1))
        if len(accs) > 1
        else (0.0 if accs else None)
    )
    runtimes = [r["runtime_seconds"] for r in runs if r["runtime_seconds"] is not None]
    mean_runtime = float(np.mean(runtimes)) if runtimes else None
    std_runtime = (
        float(np.std(runtimes, ddof=1))
        if len(runtimes) > 1
        else (0.0 if runtimes else None)
    )

    out_dir = RESULTS_ROOT / dataset / variant / model
    summary = {
        "dataset": dataset,
        "variant": variant,
        "model": model,
        "num_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "runs": runs,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_runtime_seconds": mean_runtime,
        "std_runtime_seconds": std_runtime,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["seed", "test_acc", "test_loss", "runtime_seconds", "log_file"],
        )
        writer.writeheader()
        writer.writerows(runs)

    seed_acc_map = {r["seed"]: r["test_acc"] for r in runs}
    seed_cells = [
        f"{seed_acc_map[s]:.4f}" if seed_acc_map.get(s) is not None else "—"
        for s in ordered_seeds
    ]
    mean_std_str = (
        f"{mean_acc:.4f} ± {std_acc:.4f}"
        if mean_acc is not None and std_acc is not None
        else "incomplete"
    )

    table_rows.append({
        "dataset": dataset,
        "variant": variant,
        "model": model,
        "n": str(len(accs)),
        "seed_cells": seed_cells,
        "mean_std": mean_std_str,
    })

    runtime_seed_map = {r["seed"]: r["runtime_seconds"] for r in runs}
    runtime_seed_cells = [
        f"{runtime_seed_map[s]:.1f}" if runtime_seed_map.get(s) is not None else "—"
        for s in ordered_seeds
    ]
    runtime_mean_std = (
        f"{mean_runtime:.1f} ± {std_runtime:.1f}"
        if mean_runtime is not None and std_runtime is not None
        else "no timing data"
    )
    efficiency_rows.append({
        "dataset": dataset,
        "variant": variant,
        "model": model,
        "n": str(len(runtimes)),
        "seed_cells": runtime_seed_cells,
        "mean_std": runtime_mean_std,
    })

if not table_rows:
    print("No completed train.log files found under Results/.")
else:
    headers = (
        ["Dataset", "Variant", "Model", "Seeds Done"]
        + [f"Seed {s}" for s in ordered_seeds]
        + ["Test Acc (Mean ± SD)"]
    )
    formatted = [
        [r["dataset"], r["variant"], r["model"], r["n"]]
        + r["seed_cells"]
        + [r["mean_std"]]
        for r in table_rows
    ]
    widths = [
        max(len(h), *(len(row[i]) for row in formatted))
        for i, h in enumerate(headers)
    ]

    def fmt_line(vals):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(fmt_line(headers))
    print(sep)
    for row in formatted:
        print(fmt_line(row))

    print("\nEfficiency (runtime in seconds)")
    efficiency_headers = (
        ["Dataset", "Variant", "Model", "Timed Runs"]
        + [f"Seed {s}" for s in ordered_seeds]
        + ["Runtime (Mean ± SD) s"]
    )
    efficiency_formatted = [
        [r["dataset"], r["variant"], r["model"], r["n"]]
        + r["seed_cells"]
        + [r["mean_std"]]
        for r in efficiency_rows
    ]
    efficiency_widths = [
        max(len(h), *(len(row[i]) for row in efficiency_formatted))
        for i, h in enumerate(efficiency_headers)
    ]

    def fmt_efficiency_line(vals):
        return "| " + " | ".join(
            v.ljust(efficiency_widths[i]) for i, v in enumerate(vals)
        ) + " |"

    efficiency_sep = "|-" + "-|-".join("-" * w for w in efficiency_widths) + "-|"
    print(fmt_efficiency_line(efficiency_headers))
    print(efficiency_sep)
    for row in efficiency_formatted:
        print(fmt_efficiency_line(row))
