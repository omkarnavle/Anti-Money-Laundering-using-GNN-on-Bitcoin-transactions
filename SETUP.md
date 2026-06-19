# 🚀 Setup & Reproduction Guide

> **Step-by-step instructions to run all notebooks — on Colab, Kaggle, or locally.**

---

## ⚡ Quickstart — Google Colab (Recommended)

No installation needed. Click any badge to open directly in Colab with a free GPU.

| Notebook | Description | Open |
|---|---|---|
| `GNN_10_Trials.ipynb` | GNN ablation — 10 Optuna trials | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14MobE8CAeG7Qxr0a91dIPQOS6aypFRVM?usp=sharing) |
| `GNN_30_Trials.ipynb` | GNN ablation — 30 Optuna trials | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FAj9awvhJR9SDEo8mVnHd0VfQzc7lqVn?usp=sharing) |
| `EDA_Transactions.ipynb` | Full EDA of Elliptic dataset | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16BQe7DIcMKV1SuRZOWTi2YGq7DUJj_KU?usp=sharing) |
| `ML_Wallets.ipynb` | ML pipeline on Elliptic++ wallets | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mdNa_StrgV7iQXDyVpXJCx0yrp0Eb0fl?usp=sharing) |

The notebooks auto-download the dataset via `kagglehub` — no manual download needed.

---

## 🖥️ Running on Kaggle (Best for Full GNN Training)

Kaggle provides free GPU T4 × 2 (30 hours/week) — ideal for running all 12 GNN experiments.

### Steps

1. **Create a Kaggle account** at [kaggle.com](https://kaggle.com)
2. **Verify your phone number** — Settings → Phone Verification → enter OTP  
   *(Required before GPU access is unlocked — grayed-out accelerators = unverified account)*
3. **Upload the notebook** or create a new one
4. **Enable GPU** — Settings → Accelerator → GPU T4 × 2
5. **In your first cell, install only what Kaggle doesn't pre-install:**

```python
import subprocess
subprocess.run(['pip', 'install', '-q', 'torch-geometric', 'optuna'], check=True)
# ⚠️ Do NOT reinstall torch or torchvision
# Kaggle's pre-installed PyTorch is already matched to the GPU driver
# Reinstalling causes: "CUDA error: no kernel image is available for execution"
print('Done.')
```

6. **Run all cells** — Run → Run All

### Common Kaggle Errors

| Error | Cause | Fix |
|---|---|---|
| GPU options grayed out | Phone not verified | Settings → Phone Verification |
| `CUDA error: no kernel image` | Wrong PyTorch version installed | Remove `pip install torch` — use Kaggle's pre-installed version |
| `All trials failed` | Model or data not on GPU | Re-run from top after enabling GPU |
| Out of memory | Previous experiments filled GPU RAM | Restart session — `Runtime → Restart` |

---

## 💻 Local Setup

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (strongly recommended — CPU training is very slow)
- 16 GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AML-GNN.git
cd AML-GNN

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### requirements.txt includes

```
torch >= 2.2.0
torch-geometric >= 2.4.0
optuna >= 3.5.0
pandas · numpy · scikit-learn · matplotlib · seaborn
kagglehub · networkx · scipy · xgboost · shap · tqdm
```

---

## 📥 Dataset Download

Datasets are downloaded automatically inside every notebook via `kagglehub`. If you prefer to download manually:

| Dataset | Source | Link |
|---|---|---|
| Elliptic Bitcoin Transactions | Kaggle | [ellipticco/elliptic-data-set](https://www.kaggle.com/ellipticco/elliptic-data-set) |
| Elliptic++ Wallets | Kaggle | [ellipticco/ellipticplusplus](https://www.kaggle.com/datasets/ellipticco/ellipticplusplus) |

> **Note:** `.csv` and `.pt` (checkpoint) files are excluded from this repository via `.gitignore` — they are too large for GitHub. Always download from Kaggle directly.

---

## 🔁 Reproducing Results

### GNN Experiments

| Setting | `N_TRIALS` | Runtime (T4 GPU) | Quality |
|---|---|---|---|
| Fast test | 10 | ~1.5 hours (all 12 experiments) | Approximate |
| Paper-quality | 30 | ~3–4 hours (all 12 experiments) | Reliable |
| Full replication | 100 | ~10–12 hours | Publication quality |

Change `N_TRIALS` in the experiment config cell before running:

```python
N_TRIALS = 30   # change to 100 for full paper-quality results
```

### Run Order (within each GNN notebook)

```
Cell 1  : Install packages        (torch-geometric, optuna only)
Cell 2  : Imports
Cell 3  : Download dataset
Cell 4  : Load CSVs
Cell 5  : EDA plot
Cell 6  : Graph construction + CLASS_MAP + sanity checks
Cell 7  : Class weights
Cell 8  : Model definitions       (GCN · GAT · GATv2 · GraphSAGE)
Cell 9  : Training utilities      (train_and_evaluate · evaluate_test · build_model)
Cell 10 : Experiment config       (N_TRIALS · EXPERIMENT_SETTINGS · data_device)
Cells 11–22 : Individual experiments  (one cell per model × setting)
Cell 23 : Results summary table
Cell 24 : Plots                   (AUPRC bars · ROC curves · PR curves)
Cell 25 : Save to CSV
```

> ⚠️ **Always run from Cell 1 after a session restart.** After restart, only `torch` is imported — all other variables (`data`, `class_weights`, `build_model`, `DEVICE`) are undefined until their cells are re-executed.

### Critical: CLASS_MAP

The correct label mapping must be:

```python
CLASS_MAP = {'1': 1, '2': 0, 'unknown': 2}
# Raw '1' = ILLICIT  = positive class (2.23% of all nodes) ← minority
# Raw '2' = LICIT    = negative class (20.62% of all nodes)
# 'unknown'          = excluded from masks, kept in graph
```

Swapping `'1'` and `'2'` inverts the entire task — the model would be trained to detect licit transactions instead of illicit ones, producing near-random AUPRC scores.

---

## 💾 Checkpoint Management

Each completed experiment saves a `.pt` checkpoint:

```
checkpoints/
├── GCN_baseline_best.pt
├── GCN_xavier_only_best.pt
├── GCN_graphnorm_xavier_best.pt
├── GAT_baseline_best.pt
...
└── SAGE_graphnorm_xavier_best.pt
```

### Loading saved checkpoints (to skip retraining)

```python
import torch, glob

all_results = {}
for ckpt_path in sorted(glob.glob('checkpoints/*.pt')):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # weights_only=False required on PyTorch >= 2.6
    model = build_model(ckpt['model_name'], data.num_features,
                        ckpt['hparams'], ckpt['graphnorm'], ckpt['init_mode'])
    model.load_state_dict(ckpt['state_dict'])
    test_eval = evaluate_test(model, data_device, test_mask_d, DEVICE)
    key = (ckpt['setting_name'], ckpt['model_name'])
    all_results[key] = {**test_eval, 'val_auprc': ckpt['val_auprc'],
                        'best_params': ckpt['hparams']}
    print(f"{ckpt['model_name']:8} / {ckpt['setting_name']:20} "
          f"AUPRC={test_eval['test_auprc']:.4f}")
```

> **If Colab/Kaggle session restarts**, checkpoint files are lost. Download `.pt` files to your local machine after each experiment to avoid re-running everything.

---

## 🧪 Debugging Common Issues

| Issue | Symptom | Fix |
|---|---|---|
| All Optuna trials fail | `All trials failed. Skipping.` | Run debug cell to see real error (see below) |
| Empty training curve plots | Blank axes, `No artists with labels` warning | Curves not stored in checkpoints — use checkpoint-based plotting |
| `KeyError: 'test_probs'` | ROC/PR curve cell fails | Re-run checkpoint loading cell to recompute probs |
| `ValueError: could not convert string to float: 'Unknown'` | Feature matrix contains string columns | Use `int_cols = [c for c in feat_df.columns if isinstance(c, int) and c >= 2]` |
| `AssertionError: illicit should be minority` | CLASS_MAP inverted | Fix to `{'1': 1, '2': 0, 'unknown': 2}` |
| `UnpicklingError` on `torch.load` | PyTorch >= 2.6 security change | Add `weights_only=False` to `torch.load()` |

### Debug cell — reveals hidden Optuna errors

```python
test_hparams = {'hidden_dim': 128, 'embedding_dim': 64,
                'num_layers': 2, 'dropout': 0.2}
test_model = build_model('GCN', data.num_features, test_hparams, False, 'default')
try:
    result = train_and_evaluate(
        test_model, data_device, train_mask_d, val_mask_d, test_mask_d,
        lr=0.001, n_epochs=5, weight_decay=5e-4,
        device=DEVICE, class_weights=class_weights, verbose=True,
    )
    print(f'SUCCESS — Val AUPRC: {result["best_val_auprc"]:.4f}')
except Exception as e:
    import traceback
    traceback.print_exc()
```

---

## 📋 Environment Check

Run this at the start of every session to verify everything is loaded:

```python
print('torch            :', 'torch' in dir())
print('data             :', 'data' in dir())
print('class_weights    :', 'class_weights' in dir())
print('build_model      :', 'build_model' in dir())
print('train_and_evaluate:', 'train_and_evaluate' in dir())
print('DEVICE           :', DEVICE if 'DEVICE' in dir() else 'NOT DEFINED')
# All should be True / show 'cuda' before running experiments
```

---

*For questions or issues, open a GitHub Issue in this repository.*
