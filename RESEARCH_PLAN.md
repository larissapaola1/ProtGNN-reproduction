# ProtGNN reproduction: research plan

Paper: *ProtGNN: Towards Self-Explaining Graph Neural Networks* (Zhang et al., AAAI 2022), arXiv:2112.00911

Order of work: reproduce the paper's claims first (1–3), then extend (4–5).

---

## 1. Reproduce the accuracy results (Table 1)

**Paper's claim:** ProtGNN is about as accurate as a normal GNN, and sometimes better, while also being explainable.

**What to do:**
- Datasets: MUTAG, BBBP, Graph-SST2, BA-Shape. Graph-Twitter isn't in the authors' data zip.
- Encoders: GCN, GIN, GAT, each as a plain GNN and as ProtGNN.
- Use **5–10 seeds** instead of the paper's 3. MUTAG's test set is only ~20 graphs, so 1 graph = 5 points of accuracy.
- Run two configurations, because they differ:

| Setting | Repo default | Paper |
|---|---|---|
| Cluster loss weight | 0.0 | 0.10 |
| Separation loss weight | 0.0 | 0.05 |
| Diversity loss weight | 0.00 (hard-coded) | 0.01 |
| Epochs | 800 | 500 |
| Projection period | every 10 epochs | every 50 epochs |
| MCTS rollouts | 10 | 20 |

**Things already found (from earlier test runs):**
- With the paper's separation weight, ProtGNN on BBBP collapses to predicting the majority class (≈73% vs. the paper's 89%). Turning the separation loss off fixed it on the seeds tested.
- Early stopping (80 epochs of patience) usually ends training before projection starts at epoch 100, so most "ProtGNN" runs never actually project their prototypes. Worth rerunning with more patience.
- The plain GNN baselines reproduce the paper well.

## 2. Number of prototypes (m)

**Paper's claim (Fig. 7):** accuracy rises with the number of prototypes per class and levels off at m = 5.

**What to do:**
- Rerun with m = 1, 3, 5, 7, 10 (`num_prototypes_per_class` in `Configures.py`).
- Plot accuracy against m.
- No new code needed.

## 3. Check the prototypes (interpretability)

**Paper's claim:** the learned prototypes are meaningful, human-understandable subgraphs.

**What to do:**
- After projection, draw the subgraph each prototype became.
- On MUTAG, check whether the mutagenic prototypes contain chemical groups known to cause mutations (for example NO₂ groups).
- **Extension:** hand-build prototypes from known mutagenic groups and compare them with the learned ones.

## 4. New datasets / another sector

**Goal:** test whether the method generalises beyond the paper's datasets.

**What to do:**
- Other MoleculeNet datasets (for example BACE, HIV). The code already supports them, and they download automatically.
- Another sector, for example protein graphs from the TU collection.

## 5. ProtGNN+ and its speed claim

**Paper's claim (Table 2):** the conditional subgraph sampling module makes ProtGNN+ much faster than using MCTS (632 s vs. more than 2 hours on BBBP).

**Problem:** the repo has no working ProtGNN+ training code. The sampling layers are defined in `train_gnns.py` but never used.

**What to do:**
- Implement ProtGNN+ from the paper's description. This is the largest task on the list.
- Meanwhile, measure ProtGNN's MCTS projection time, which is half of the comparison.
- If not implemented, report it as **not reproducible from the released code**.

---

## Priorities

| # | Experiment | New code needed? | Priority |
|---|---|---|---|
| 1 | Table 1 accuracy | No | Must do |
| 2 | Number of prototypes | No | High |
| 3 | Prototype visualisation | Small (plotting) | High |
| 4 | New datasets | Minimal | If time |
| 5 | ProtGNN+ / speed | Yes, a lot | If time |