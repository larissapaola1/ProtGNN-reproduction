> **Reproduction copy.** This is a reproduction of ProtGNN for Graph Deep Learning.
> Original repo: https://github.com/zaixizhang/ProtGNN
# ProtGNN experimental settings and reproduction changes

This README first lists the settings reported in the paper, then explains how this reproduction applies them. Line numbers refer to the current code and may shift after edits.

Paper: [ProtGNN: Towards Self-Explaining Graph Neural Networks](https://arxiv.org/pdf/2112.00911)

## 1. Experimental settings reported in the paper

### Shared settings

- **GNN backbones:** GCN, GAT, and GIN.
- **Datasets:** MUTAG, BBBP, Graph-SST2, Graph-Twitter, and BA-Shape. BA-Shape is a node-classification task; the other listed datasets are graph-classification tasks.
- **Train / validation / test split:** 80% / 10% / 10%.
- **Repeated runs:** 3 independent runs using random data splits; results are reported as mean and standard deviation.
- **Optimizer:** Adam, learning rate `0.005`.
- **Training duration:** up to 500 epochs, with early stopping based on validation accuracy. The paper does not specify the early-stopping patience.
- **Hardware reported:** one Tesla V100 GPU.

### Original GNN

- Standard GNN classifier using the selected backbone (GCN, GAT, or GIN).
- No prototype layer or prototype losses: cluster (`Clst`), separation (`Sep`), or diversity (`Div`).

### ProtGNN

- Prototype layer enabled; `m = 5` prototypes per class.
- Loss weights: `λ₁ = 0.10` for cluster loss, `λ₂ = 0.05` for separation loss, and `λ₃ = 0.01` for diversity loss.
- Diversity threshold: `s_max = 0.3`.
- Prototype projection: starts at epoch `T_p = 100`, then repeats every `τ = 50` epochs.
- MCTS projection: `λ = 5`, 20 iterations, at most 10 child expansions per node, and `N_min = 5`.

### ProtGNN+

- Includes the ProtGNN prototype and projection settings, plus the conditional subgraph-sampling module.
- Sampling warm-up: `T_w = 200` epochs.
- Edge-weight learner: three-layer neural network.
- Budget penalty weight: `λ_b = 0.01`; subgraph budget: `B = 10`.
- The paper says SGD is used to optimize the sampling objective in Eq. (14).

## 2. Changes made in this reproduction: how we did it

1. **Diversity-loss weight and threshold** (`models/train_gnns.py:208–217`)
   - **BEFORE:** The diversity-loss multiplier was `0.00`.
   - **FIX:** Changed it to `0.01` to match the paper; the threshold `s_max = 0.3` is hard-coded in the same block.

2. **Prototype projection schedule** (`Configures.py:96`; `models/train_gnns.py:166–168`)
   - **BEFORE:** Projection used the old schedule.
   - **FIX:** Start at epoch 100 and attempt projection every 50 epochs afterward.

3. **Original baseline projection** (`models/train_gnns.py:166–168`)
   - **BEFORE:** Original runs could enter ProtGNN’s prototype-projection block.
   - **FIX:** Require `model_args.enable_prot`, so Original skips prototype projection.

4. **MCTS projection settings** (`Configures.py:59–65`; `my_mcts.py:108–109`)
   - **BEFORE:** The code used 10 rollouts.
   - **FIX:** Set 20 iterations, `c_puct=5` (the paper’s MCTS `λ`), up to 10 child expansions, and `N_min=5`; the rollout count controls the MCTS loop.

5. **Seeded graph-classification splits** (`Results/run_experiments.py:78`; `models/train_gnns.py:115–119`)
   - **BEFORE:** The loader used its default split seed of 5 for every run.
   - **FIX:** Pass the selected run seed to `get_dataloader`, so each recorded seed determines its split.

6. **Per-run checkpoints** (`Results/run_experiments.py:81`; `models/train_gnns.py:125–129, 150–158`)
   - **BEFORE:** Runs for the same dataset could overwrite each other’s checkpoints.
   - **FIX:** Set and create a separate checkpoint folder for each run, with the dataset folder as a fallback for standalone training.

7. **Graph-classification test metrics** (`models/train_gnns.py:260–268`)
   - **BEFORE:** `train_GC` printed test results but returned no metrics to the runner.
   - **FIX:** Return test accuracy and loss so `run_experiments.py` can store them in summaries.

8. **Early-stopping patience** (`Configures.py:90`)
   - **BEFORE:** Patience was 80 epochs.
   - **FIX:** Set it to 100 to give training a chance to reach the first projection at epoch 100. The paper does not specify patience, so this is a reproduction choice.

9. **Maximum training epochs** (`Results/run_experiments.py:56–59`)
   - **BEFORE:** The runner could use a different epoch limit.
   - **FIX:** Set `train_args.max_epochs = 500`, matching the paper’s maximum; early stopping can still end a run sooner.

10. **Variant configuration** (`Results/run_experiments.py:30–53`)
   - **BEFORE:** The runner did not consistently select the intended Original and ProtGNN settings.
   - **FIX:** Set `enable_prot=False` for Original and `enable_prot=True` for ProtGNN; set ProtGNN+ warm-up to 200 epochs. ProtGNN+ sampling itself remains unimplemented, as described below.

11. **Per-run training runtime** (`Results/run_experiments.py:82–90, 104–108, 112–130`)
   - **BEFORE:** Elapsed runtime was not recorded in each run’s results.
   - **FIX:** Time the full `train_GC` call and save per-seed runtime and its mean.

12. **Efficiency-results table** (`Results/collect_results.py:29–48, 63–71, 88–119, 141–158, 188–214`)
   - **BEFORE:** The collector printed only accuracy results.
   - **FIX:** Read stored runtimes and print a runtime table below the accuracy table; older runs without timing data show “no timing data.”

13. **t-SNE dependency** (`Results/tsne.py:6, 126`)
   - **BEFORE:** Running t-SNE failed if scikit-learn was not installed.
   - **FIX:** Install it with `pip install scikit-learn`. It still needs to be added to `requirements.txt` for repeatable setup on another machine.

14. **Environment compatibility** (details in `REPRODUCTION_LOG.md`)
   - **BEFORE:** The released code hit compatibility/device issues with the local PyTorch/PyG versions and CPU setup.
   - **FIX:** Use the CPU/device adjustments, remove the unused PyG `topk` import, and create the missing log folder, as recorded in the reproduction log.

15. **MCTS projection candidate handling** (`models/train_gnns.py:172, 180, 184–185`)
   - **BEFORE:** The best score began at zero and `proj_prot` could remain unset, causing an `UnboundLocalError` when no score beat zero.
   - **FIX:** Initialize the score to negative infinity and `proj_prot` to `None`, ignore non-finite scores, and raise a clear error if no valid candidate exists before assignment.

16. **Original BA-Shape loss** (`models/train_gnns.py:381–402`)
   - **BEFORE:** Original BA-Shape training accessed `prototype_class_identity` and calculated prototype losses even though the Original GAT/GIN node models have no prototype layer, causing an `AttributeError` (or a distance-related `TypeError`).
   - **FIX:** Put both the prototype-class lookup and all prototype-loss calculations inside `if model_args.enable_prot:`. Original skips that block and trains with cross-entropy alone.

17. **Node-classification model return values** (`models/__init__.py:73–83`)
   - **BEFORE:** The wrapper expected four values, but GAT/GIN node-classification models return three (logits, probabilities, embeddings), causing an unpacking `ValueError`.
   - **FIX:** Accept either three or four values and use an empty prototype-distance list when the model returns three, allowing the Original node-classification path to continue.

## 3. Reproduction limits and deviations

- The runner currently uses three seeds like the paper. Randomness might give different results.
- `Results/run_experiments.py` runs only the dataset, backbone, and variants selected in its constants. Its current selection is `MUTAG`, `GCN`, and `ProtGNN`; edit the constants to run other combinations.
- **ProtGNN+ is not fully implemented.** `ModelParser` and `TrainParser` do not define active ProtGNN+ sampling flags, and `train_GC` does not run the conditional subgraph-sampling training loop. The `hasattr` checks in the runner do not activate a real module; `warm_epochs=200` alone does not reproduce ProtGNN+.
- The paper’s three-layer edge learner, SGD update for Eq. (14), and budget constraint are not wired into `train_GC`. Do not report ProtGNN+ accuracy or timing as reproduced until that training path is implemented.
- Check each ProtGNN log for “Projection of prototype completed.” If early stopping ends a run before epoch 100, its prototypes were not projected.
- Runtime currently measures the full `train_GC` call, including data loading, training, projection, test evaluation, and checkpoint writing. Use the same timing method and machine for every method you compare.
