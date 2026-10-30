# ProtGNN reproduction log

Paper: *ProtGNN: Towards Self-Explaining Graph Neural Networks* (Zhang et al., AAAI 2022), arXiv:2112.00911
Original code: https://github.com/zaixizhang/ProtGNN
My copy: https://github.com/larissapaola1/ProtGNN-reproduction

---

## 1. Environment

| Item | Version |
|---|---|
| Machine | MacBook Pro (Apple silicon), CPU only |
| Python | 3.13 (virtual environment in `.venv`) |
| PyTorch | 2.14.1 |
| PyTorch Geometric | 2.8.1 |
| RDKit | 2026.9.1 |

The repo's README asks for PyTorch 1.8.0 and PyTorch Geometric 2.0.2 (2021). These versions are no longer practical to install, so I used current versions and fixed the code where it broke.

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Data

- Downloaded the authors' dataset zip from the SharePoint link in the README.
- Placed the contents at `ProtGNN/datasets/<name>/raw/` (the path the code expects).
- Contents: `BA_2Motifs`, `BA_shapes`, `bbbp`, `Graph-SST2`, `MUTAG`.
- **Graph-Twitter is not in the zip**, so that row of the paper's Table 1 can't be reproduced from the authors' data.
- BBBP loaded with 2039 graphs, 24.06 avg. nodes, 25.95 avg. edges, matching the paper's Table 3.

## 3. Errors found and fixes

Command used each time: `python -m models.train_gnns` (repo defaults: GCN on BBBP).

### Error 1: `topk` import fails

```
File "models/GCN.py", line 6, in <module>
    from torch_geometric.nn.pool.topk_pool import topk
ImportError: cannot import name 'topk' from 'torch_geometric.nn.pool.topk_pool'
```

**Cause:** newer PyTorch Geometric no longer provides `topk` at that location. The code crashes while importing, before any training.

**Check:** searched `GCN.py` for `topk`; it appears only on the import line and is never used.

**Fix:** deleted the unused import in `models/GCN.py`.

Before:
```python
from torch_geometric.nn.pool.topk_pool import topk
```
After: line removed.

---

### Error 2: tensors on different devices

```
File "models/train_gnns.py", line 180, in train_GC
    prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y].bool()).to(model_args.device)
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

**Cause:**
- `Configures.py` sets `self.device = 0` ("device number 0"). On a Mac with current PyTorch, this means the Apple GPU (`mps`), so the data, including the labels `batch.y`, was moved there.
- `prototype_class_identity` (created in `GCN.py`) is a plain tensor attribute, not a registered parameter, so it isn't moved with the model and stays on the CPU.
- PyTorch won't index a CPU tensor with indices that live on the GPU.
- The authors used an NVIDIA GPU with older PyTorch, which apparently allowed this.

**Fix:** run everything on the CPU, so all tensors are on the same device. The models are small, so CPU is fast enough.

`Configures.py`, line 30:
```python
# before
self.device: int = 0
# after
self.device = torch.device('cpu')
```

`models/GCN.py`, lines 31 and 148 (these hard-coded an NVIDIA device and would build the invalid name `'cuda:cpu'` after the change above):
```python
# before
self.device = torch.device('cuda:'+str(model_args.device))
# after
self.device = model_args.device
```

---

### Error 3: missing `log` folder

```
File "models/train_gnns.py", line 35, in append_record
    f = open('./log/hyper_search', 'a')
FileNotFoundError: [Errno 2] No such file or directory: './log/hyper_search'
```

**Cause:** the script writes each epoch's results to `log/hyper_search`. `open(..., 'a')` can create the file but not the missing folder. Git doesn't store empty folders, so the authors' `log` folder never reached the repo.

**Progress at this point:** the model built, the data loaded and one full epoch trained. It only failed when writing the log.

**Fix:**
```bash
mkdir log
```

---

### Result: training runs

After these fixes, `python -m models.train_gnns` trains end to end on the CPU.

## 4. Warnings still printed (not errors)

These don't stop the code; they are notices about outdated functions:
- `FutureWarning: torch.jit.script is deprecated` (comes from inside PyTorch)
- `UserWarning: It is not recommended to directly access the internal storage format data` (`load_dataset.py`, lines 325 and 327)
- `UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead` (`load_dataset.py`, lines 371–373)

## 5. Notes for the report

- The released code does not run with current library versions without changes: it fails at import time.
- It also assumes an NVIDIA GPU (hard-coded `cuda` device).
- The fixes above change only compatibility and hardware, not the model or the training method.
- The repo's default settings differ from the paper's (to be checked next): cluster/separation loss weights 0.0 (paper: 0.10 / 0.05), diversity loss multiplied by 0.00 (paper: 0.01), 800 epochs (paper: 500), projection every 10 epochs (paper: 50), 10 MCTS rollouts (paper: 20).