
# 🧠 nn-checkpoint

**ZIP-powered neural network checkpoints** • Keeps **only the best N** • Framework-agnostic • Tiny • Blazing fast • Production-ready

[![PyPI version](https://img.shields.io/pypi/v/nn-checkpoint.svg)](https://pypi.org/project/nn-checkpoint/)
[![Python](https://img.shields.io/pypi/pyversions/nn-checkpoint)](https://pypi.org/project/nn-checkpoint/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yourusername/nn-checkpoint/blob/main/LICENSE)
[![Tests](https://github.com/yourusername/nn-checkpoint/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/nn-checkpoint/actions)
[![Downloads](https://img.shields.io/pypi/dm/nn-checkpoint)](https://pypi.org/project/nn-checkpoint/)

---

### First-Principles Design

A perfect checkpoint must satisfy these axioms:
- Survive crashes & machine transfers
- Occupy minimal disk (no 100+ junk files)
- Be instantly inspectable
- Load in <100 ms
- Work with **any** framework (PyTorch, JAX, Flax, pure NumPy, TensorFlow…)
- Never force you to keep every epoch

**nn-checkpoint** is the first library built exactly around those axioms.

---

### ✨ Features

- Single self-contained `.zip` with:
  - `architecture.json`
  - Custom ultra-compressible `weights.bin` (hand-crafted binary format)
  - `optimizer.pkl` (optional)
  - `metadata.json` + auto timestamp
  - Full `training_log.csv` (pandas-ready)
- `CheckpointManager` — automatically keeps only the **top N** checkpoints by any metric (`val_loss`, `val_acc`, …) and prunes the rest
- One-line PyTorch / JAX integration
- CLI: `nn-inspect` — peek inside any checkpoint without loading the model
- Zero heavy dependencies (only `numpy`)

---

### 📦 Installation

```bash
pip install nn-checkpoint
```

**Dev / editable install:**
```bash
git clone https://github.com/yourusername/nn-checkpoint.git
cd nn-checkpoint
pip install -e ".[dev]"
```

---

### 🚀 Quick Start

```python
from nn_checkpoint import Checkpoint, CheckpointManager
import numpy as np

# Your model (any framework)
architecture = {
    "layers": [
        {"type": "Linear", "in": 784, "out": 256, "activation": "relu"},
        {"type": "Linear", "in": 256, "out": 10, "activation": "softmax"},
    ]
}

state_dict = {
    "fc1.weight": np.random.randn(256, 784).astype(np.float32),
    "fc1.bias":   np.random.randn(256).astype(np.float32),
    # ...
}

manager = CheckpointManager("checkpoints/", keep_top_n=5, metric="val_loss")

ckpt = Checkpoint(
    architecture=architecture,
    state_dict=state_dict,
    optimizer_state=None,           # or your optimiser dict
    metadata={"task": "mnist", "run_id": "exp-42"}
)

for epoch in range(1, 51):
    # ... your training step ...
    ckpt.state_dict = new_weights          # update
    ckpt.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        lr=lr
    )

    saved_path = manager.save(ckpt, epoch)   # ← only saves if top-5 !
    print(f"Epoch {epoch:02d} → {'✅ saved' if saved_path else '⏭ skipped'}")

# Resume / evaluate
best = manager.best()          # loads the absolute best
print(best.summary())

latest_epoch = manager.latest_epoch()
print(f"Ready to resume from epoch {latest_epoch}")
```

See full demo: [`examples/train_loop.py`](examples/train_loop.py)

---

### 🔧 Core API

```python
# Save/Load
ckpt.save("best.zip")
loaded = Checkpoint.load("best.zip")

# Manager
manager = CheckpointManager(
    directory="runs/exp001/",
    keep_top_n=3,
    metric="val_acc",      # or "val_loss"
    mode="max"
)

path = manager.save(ckpt, epoch)
best_ckpt = manager.best()
```

---

### 🛠 CLI Inspection

```bash
# Installs automatically with the package
nn-inspect checkpoints/epoch_0042.zip

# Output includes:
# • Raw ZIP compression stats
# • Tensor count & total parameters
# • Last 5 epochs table
# • Best epoch highlight
```

---

### Why This Beats `torch.save`, `save_pretrained`, SafeTensors…

| Feature                  | nn-checkpoint | torch.save | HF save_pretrained | SafeTensors |
|--------------------------|---------------|------------|--------------------|-------------|
| Keeps only top-N         | ✅ Auto       | ❌ Manual   | ❌ Manual           | ❌          |
| Full training log CSV    | ✅ Built-in   | ❌         | Partial            | ❌          |
| Human-readable arch + meta | ✅           | ❌         | ✅                 | ❌          |
| Custom fast binary weights | ✅ Excellent compression | Pickle bloat | ✅ | ✅ |
| Zero-framework lock-in   | ✅ NumPy only | PyTorch only | HF only          | Generic    |
| CLI peek tool            | ✅ `nn-inspect` | ❌       | ❌                 | ❌          |
| Disk-friendly by design  | ✅            | ❌         | ❌                 | Partial    |

---

### 📁 Project Layout

```
nn-checkpoint/
├── nn_checkpoint/          # import nn_checkpoint
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── manager.py
│   └── inspect.py
├── examples/
├── tests/
├── pyproject.toml          # modern packaging
├── README.md               # ← you are here
└── LICENSE (MIT)
```

---

### 🧪 Testing & Development

```bash
pip install -e ".[dev]"
pytest
black .
ruff check .
```

---

### Contributing • Roadmap • Ideas

PRs welcome!  
Planned next:
- Official PyTorch & JAX helper classes
- Sharded/multi-GPU support
- Built-in quantization + ONNX export
- GitHub Actions + automatic PyPI release

---

**Made with 🔥 and first-principles reasoning.**  
Star ⭐ the repo if this saves you from checkpoint hell.

---

**License** — MIT  
**Author** — James Squire 
**Version** — 0.2.0 (March 2026)

---

