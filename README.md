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

1. **Matched the diversity settings.** In `models/train_gnns.py:208–217`, We changed the diversity-loss multiplier from `0.00` to `0.01`; the threshold `0.3` is currently hard-coded in the same block.

2. **Matched the projection schedule.** `Configures.py:96` sets the start epoch to 100, and `models/train_gnns.py:166–168` attempts projection every 50 epochs after that.

3. **Stopped Original runs from projecting prototypes.** The condition at `models/train_gnns.py:166–168` also checks `model_args.enable_prot`, so the Original baseline skips this ProtGNN step.

4. **Matched the MCTS iteration count.** `Configures.py:59–65` sets 20 rollouts, `c_puct=5`, `expand_atoms=10`, and `min_atoms=5`; `my_mcts.py:108–109` uses the rollout count as the loop iteration count.

5. **Made the split follow the recorded seed.** `Results/run_experiments.py:78` selects each seed, and `models/train_gnns.py:115–119` passes it into `get_dataloader`, instead of using the loader’s default split seed of 5 for every run.

6. **Saved each run’s checkpoints separately.** `Results/run_experiments.py:81` sets a per-seed `save_dir`; `models/train_gnns.py:125–129, 150–158` uses and creates it, with the dataset checkpoint folder as a fallback for standalone runs.

7. **Returned test metrics to the runner.** `models/train_gnns.py:260–268` returns test accuracy and loss, allowing `Results/run_experiments.py` to put accuracy in its summaries.

8. **Set early-stopping patience to 100.** `Configures.py:90` changes the code default from 80 to 100 to give ProtGNN a chance to reach its first projection at epoch 100. This patience is a reproduction choice because the paper does not state a value.

9. **Set the experiment runner’s maximum to 500 epochs.** `Results/run_experiments.py:56–59` sets `train_args.max_epochs = 500`; early stopping may end a run sooner.

10. **Configured the model variants.** `Results/run_experiments.py:30–53` sets `enable_prot=False` for Original and `enable_prot=True` for ProtGNN; it sets `warm_epochs=200` for ProtGNN+ to match the reported warm-up value.

11. **Recorded elapsed runtime.** `Results/run_experiments.py:82–90, 104–108` times the full `train_GC` call and stores seconds per seed; lines 112–130 save the mean and per-run values.

12. **Added an efficiency-results table.** `Results/collect_results.py:29–48, 63–71, 88–119, 141–158, 188–214` reads stored runtimes, preserves them in summaries, and prints a runtime table below the accuracy table. Older runs without timing data show “no timing data.”

13. **Added the local t-SNE dependency.** `Results/tsne.py:6, 126` imports and uses scikit-learn; it was installed with `pip install scikit-learn`. Add it to `requirements.txt` to make the setup repeatable on another machine.

14. **Applied environment compatibility fixes.** The CPU/device adjustment, removal of an unused PyG `topk` import, and creation of the missing log folder are recorded in `REPRODUCTION_LOG.md`.

15. **Made prototype projection handle missing or invalid MCTS scores.** In `models/train_gnns.py:172`, we initialize the best score to negative infinity and `proj_prot` to `None`, so a finite candidate with a zero or negative score can still be selected. At line 180, we ignore non-finite scores; at lines 184–185, we raise a clear error if no valid candidate was found before assigning the projected prototype. This prevents the previous `UnboundLocalError` and makes invalid MCTS output easier to diagnose.

## 3. Reproduction limits and deviations

- The runner currently uses three seeds like the paper. Randomness might give different results.
- `Results/run_experiments.py` runs only the dataset, backbone, and variants selected in its constants. Its current selection is `MUTAG`, `GCN`, and `ProtGNN`; edit the constants to run other combinations.
- **ProtGNN+ is not fully implemented.** `ModelParser` and `TrainParser` do not define active ProtGNN+ sampling flags, and `train_GC` does not run the conditional subgraph-sampling training loop. The `hasattr` checks in the runner do not activate a real module; `warm_epochs=200` alone does not reproduce ProtGNN+.
- The paper’s three-layer edge learner, SGD update for Eq. (14), and budget constraint are not wired into `train_GC`. Do not report ProtGNN+ accuracy or timing as reproduced until that training path is implemented.
- Check each ProtGNN log for “Projection of prototype completed.” If early stopping ends a run before epoch 100, its prototypes were not projected.
- Runtime currently measures the full `train_GC` call, including data loading, training, projection, test evaluation, and checkpoint writing. Use the same timing method and machine for every method you compare.
