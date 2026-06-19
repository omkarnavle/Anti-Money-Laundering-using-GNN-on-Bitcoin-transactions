# Setup & Reproduction Guide

## Quickstart (Google Colab — Recommended)

All notebooks run on Google Colab with a free T4 GPU. No local setup needed.

| Notebook | Direct Link |
|---|---|
| GNN — 10 Trials | https://colab.research.google.com/drive/14MobE8CAeG7Qxr0a91dIPQOS6aypFRVM |
| GNN — 30 Trials | https://colab.research.google.com/drive/1FAj9awvhJR9SDEo8mVnHd0VfQzc7lqVn |
| Dataset EDA | https://colab.research.google.com/drive/16BQe7DIcMKV1SuRZOWTi2YGq7DUJj_KU |
| ML — Wallets | https://colab.research.google.com/drive/1mdNa_StrgV7iQXDyVpXJCx0yrp0Eb0fl |

## Running on Kaggle (GPU — Recommended for Full Training)

1. Create a Kaggle account and verify your phone number (required for GPU access)
2. Go to **Settings → Accelerator → GPU T4 x2**
3. In the first cell, install only the missing packages:

```python
import subprocess
subprocess.run(['pip', 'install', '-q', 'torch-geometric', 'optuna'], check=True)
# Do NOT reinstall torch — use Kaggle's pre-installed version
```

4. Run all cells top to bottom

## Dataset Download

The datasets are downloaded automatically inside each notebook via `kagglehub`. If running locally, you can also download manually:

- **Elliptic Transactions:** https://www.kaggle.com/ellipticco/elliptic-data-set
- **Elliptic++ Wallets:** https://www.kaggle.com/datasets/ellipticco/ellipticplusplus

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/AML-GNN.git
cd AML-GNN
pip install -r requirements.txt
jupyter notebook
```

> **Note:** Full-graph GNN training on CPU is very slow. A GPU is strongly recommended.

## Reproducing Results

### GNN Results (Table 2 equivalent)

Set `N_TRIALS = 100` in the experiment config cell for full paper-quality hyperparameter search. The notebooks default to 30 trials for speed.

### Expected Runtime

| Experiment | GPU (T4) | CPU |
|---|---|---|
| 1 GNN experiment (30 trials) | ~15–20 min | ~3–4 hours |
| All 12 GNN experiments | ~3–4 hours | Not recommended |
| ML Wallets notebook | ~10 min | ~10 min |
