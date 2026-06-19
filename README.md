# 🔗 Anti-Money Laundering Using Graph Neural Networks

> **Detecting illicit Bitcoin transactions using Graph Neural Networks on transaction-level data  
> and Machine Learning on wallet-level data — combining two complementary detection layers.**

---

## 📌 Project Overview

This project is an end-to-end research implementation built as part of the **MSc in Statistics and Data Science** programme at **SVKM's Narsee Monjee Institute of Management Studies (NMIMS), Mumbai**.

The project applies Graph Neural Networks to the Elliptic Bitcoin Dataset, conducting a systematic ablation study of Xavier initialisation and GraphNorm across four architectures — while simultaneously building a wallet-level ML fraud detection pipeline using the Elliptic++ dataset.

| Metric | Value |
|---|---|
| Transaction Nodes | 203,769 |
| Payment Edges | 234,355 |
| Features per Transaction | 165 |
| Illicit Transactions | 4,545 (2.23% of all nodes) |
| GNN Architectures Tested | 4 (GCN · GAT · GATv2 · GraphSAGE) |
| Training Settings per Model | 3 (Baseline · Xavier · GraphNorm+Xavier) |
| Total GNN Experiments | 12 |
| Wallet Addresses (ML Track) | 822,942 |
| ML Models Trained | 4 (RF · XGBoost · MLP · LR) |
| Best GNN AUPRC | 0.534 (GATv2 · Xavier Only) |
| Best ML MCC | 0.856 (Random Forest) |

---

## 👥 Authors

| Name | Roll No. | Enrollment No. |
|---|---|---|
| Arslan Anzar | A004 | 86062500041 |
| Ruchika Basutkar | A005 | 86062500052 |
| Parin Kulkarni | A029 | 86062500035 |
| Omkar Navle | A037 | 86062500050 |
| Diya Rawat | A048 | 86062500030 |

**Mentor:** Rutik Bhurke  
**Institution:** Nilkamal School of Mathematics, Applied Statistics and Analytics  
**Programme:** MSc Statistics and Data Science  
**Date:** 16th April 2026

---

## 📂 Repository Structure

```
AML-GNN/
│
├── notebooks/
│   ├── GNN_10_Trials.ipynb            ← GNN ablation — 4 models × 3 settings, 10 Optuna trials
│   ├── GNN_30_Trials.ipynb            ← GNN ablation — 4 models × 3 settings, 30 Optuna trials
│   ├── EDA_Transactions.ipynb         ← Full exploratory data analysis of Elliptic dataset
│   └── ML_Wallets.ipynb               ← ML pipeline on Elliptic++ wallet addresses
│
├── reports/
│   ├── Final_Report.docx              ← Full academic project report
│   └── Presentation.pptx             ← Project presentation slides
│
├── assets/
│   └── images/                        ← Slide exports used in this README
│
├── docs/
│   └── methodology.md                 ← Deep-dive on all design decisions
│
├── requirements.txt                   ← Python dependencies
├── SETUP.md                           ← Reproduction guide (Colab · Kaggle · Local)
├── .gitignore
└── README.md                          ← This file
```

---

## 📓 Notebooks

| # | Notebook | Description | Open in Colab |
|---|---|---|---|
| 1 | `GNN_10_Trials.ipynb` | GNN ablation — 10 Optuna trials per experiment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14MobE8CAeG7Qxr0a91dIPQOS6aypFRVM?usp=sharing) |
| 2 | `GNN_30_Trials.ipynb` | GNN ablation — 30 Optuna trials per experiment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FAj9awvhJR9SDEo8mVnHd0VfQzc7lqVn?usp=sharing) |
| 3 | `EDA_Transactions.ipynb` | Full EDA — graph structure, heterophily, temporal drift | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16BQe7DIcMKV1SuRZOWTi2YGq7DUJj_KU?usp=sharing) |
| 4 | `ML_Wallets.ipynb` | Wallet-level ML — RF, XGBoost, MLP, LR with SHAP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mdNa_StrgV7iQXDyVpXJCx0yrp0Eb0fl?usp=sharing) |

---

## 🗂️ Datasets

### Track 1 — Elliptic Bitcoin Transaction Dataset

**Source:** Weber et al. (2019) · [Kaggle — Elliptic Data Set](https://www.kaggle.com/ellipticco/elliptic-data-set)

| Attribute | Detail |
|---|---|
| Total nodes | 203,769 Bitcoin transactions |
| Total edges | 234,355 payment flows |
| Features | 165 (94 local transaction + 71 aggregated neighbourhood) |
| Time steps | 49 (~2 weeks each, Jan 2009 – Sep 2017) |
| Illicit nodes | 4,545 — 2.23% of all nodes / 9.8% of labeled |
| Licit nodes | 42,019 — 20.62% of all nodes / 90.2% of labeled |
| Unknown nodes | 157,205 — 77.15% (kept in graph, excluded from masks) |
| Key label mapping | Raw `'1'` = **illicit** (positive class) · Raw `'2'` = **licit** (negative class) |

### Track 2 — Elliptic++ Wallets Dataset

**Source:** Elmougy & Liu, KDD '23 · [Kaggle — Elliptic++](https://www.kaggle.com/datasets/ellipticco/ellipticplusplus)

| Attribute | Detail |
|---|---|
| Total wallet addresses | 822,942 |
| Features per address | 56 |
| Time steps | 49 |
| Illicit wallets | 1.73% |
| Data files | `wallets_features.csv` · `wallets_classes.csv` · `AddrAddr_edgelist.csv` |

---

## 🔬 Track 1 — GNN Architecture & Methodology

![Slides Overview — Title, Dataset, Methodology, GNN Architecture](assets/images/slides_1.jpg)

### Models Evaluated

| Model | Key Mechanism | PyG Layer |
|---|---|---|
| **GCN** | Degree-normalised neighbourhood averaging | `GCNConv` |
| **GAT** | Static learned attention weights per neighbour | `GATConv` |
| **GATv2** | Dynamic attention — strictly more expressive than GAT | `GATv2Conv` |
| **GraphSAGE** | Inductive sampling + mean/max aggregation | `SAGEConv` |

### Ablation Study — 3 Settings × 4 Models = 12 Experiments

| Setting | GraphNorm | Weight Initialisation |
|---|---|---|
| `baseline` | ❌ | Default (Kaiming) |
| `xavier_only` | ❌ | Xavier Uniform |
| `graphnorm_xavier` | ✅ | Xavier Uniform |

### Key Design Decisions

**Temporal train/val/test split — not random:**

```
Train  :  time steps  1 – 29   (first ~15 months)
Val    :  time steps 30 – 39
Test   :  time steps 40 – 49   (future fraud — never seen during training)
```

Random splits allow future fraud patterns to "teach" the model — temporal splits reflect real deployment where future fraud is genuinely unknown.

**Weighted CrossEntropyLoss:**
```
w_illicit  =  n_total / (2 × n_illicit)   →  ~5.1×  upweight  (minority)
w_licit    =  n_total / (2 × n_licit)     →  ~0.55× downweight (majority)
```
Without weighting, the model learns to predict "all licit" and achieves 90.2% accuracy while catching zero fraudsters.

**AUPRC as primary metric:**
A random classifier achieves AUPRC ≈ 9.8% (= positive rate). Models at AUPRC > 0.50 represent a 5× improvement over random. Standard accuracy is meaningless on a 90/10 imbalanced split.

**Optuna TPE hyperparameter search:**
Bayesian optimisation using Tree-structured Parzen Estimator — searches `hidden_dim`, `embedding_dim`, `num_layers`, `dropout`, `lr`, `n_epochs`, `weight_decay`.

---

## 📊 GNN Results

![GNN Variants, Results Comparison, Training Curves](assets/images/slides_2.jpg)

### 10 Trials vs 30 Trials

| Model | Best Setting | AUPRC (n=10) | AUROC (n=10) | AUPRC (n=30) | AUROC (n=30) |
|---|---|---|---|---|---|
| GCN | Baseline | 0.441 | 0.804 | 0.441 | 0.804 |
| GAT | Baseline | 0.233 | 0.811 | 0.531 | 0.811 |
| **GATv2** | **Xavier Only** | **0.499** | **0.821** | **0.534** | **0.820** |
| SAGE | Baseline | 0.492 | 0.815 | 0.521 | 0.834 |

**Key findings:**

- **GATv2 achieves the highest AUPRC (0.534)** with stable training — val loss tracks train loss with no divergence across epochs
- **n=30 reliably outperforms n=10** — wider Optuna search finds more robust hyperparameter configurations rather than lucky ones
- **Xavier initialisation consistently helps** — GATv2 gains +0.035 AUPRC over baseline with Xavier only
- **GraphNorm benefits are architecture-dependent** — benefits vary; does not uniformly improve all models

---

## 🤖 Track 2 — ML Wallet Detection

![Key Insights, Wallet Analysis, ML Workflow, Feature Importance](assets/images/slides_3.jpg)

### Pipeline

```
Data Preprocessing  →  Stratified Split (60/20/20)  →  MinMax Scaling
→  Class Imbalance Handling  →  Model Training  →  Stratified K-Fold CV (k=5)
→  Hyperparameter Tuning  →  Feature Importance (MDI + Permutation + Drop-Column)
→  Retrain on Top 19 Features  →  Final Evaluation + SHAP
```

### Results (All 56 Features)

| Model | Precision | Recall | F1 | MCC | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.211 | 0.687 | 0.323 | 0.323 | 0.849 |
| **Random Forest** | **0.958** | **0.776** | **0.857** | **0.856** | **0.987** |
| MLP | 0.777 | 0.391 | 0.520 | 0.535 | 0.962 |
| XGBoost | 0.660 | 0.946 | 0.777 | 0.777 | 0.972 |

> **Why MCC?** Matthews Correlation Coefficient accounts for all four cells of the confusion matrix (TP, TN, FP, FN). A model predicting all-licit achieves MCC ≈ 0, immediately exposing it — unlike accuracy (which would show 98%+).

### Feature Selection — Three-Method Agreement

Top 19 of 56 features selected where **MDI + Permutation + Drop-Column all agree**:

```
Fee features     : fees_min · fees_max · fees_mean · fees_median · fees_total
                   fees_as_share_min/max/mean/median/total
Timing features  : first_sent_block · last_block_appeared_in
                   first_block_appeared_in · first_received_block
Volume features  : btc_received_median · btc_sent_max · btc_sent_total · btc_transacted_max
Network          : transacted_w_address_total · total_txs
```

> Fee and timing features are the strongest predictors of illicit wallet behaviour — confirmed by SHAP analysis across both Random Forest and XGBoost.

---

## 💡 Key Findings

![Feature Refined Results, Conclusions, Future Work](assets/images/slides_4.jpg)

| Finding | Detail |
|---|---|
| Best GNN model | GATv2 with Xavier Only — AUPRC 0.534, AUROC 0.820 |
| Best ML model | Random Forest — F1 0.857, MCC 0.856 |
| n=30 vs n=10 trials | Significantly more reliable — models learn real patterns, not lucky configs |
| Temporal split is critical | Random splits inflate results by leaking future fraud into training |
| Most important wallet features | Fee-related and block timing features (confirmed by SHAP) |
| Xavier benefit | Consistent improvement for GATv2; architecture-dependent for others |
| GraphNorm | Does not uniformly help — architecture-dependent |
| Class imbalance impact | Weighted loss is essential — without it models default to predicting all-licit |

---

## 🛠️ Tools and Technologies

| Tool | Purpose |
|---|---|
| **PyTorch + PyTorch Geometric** | GNN model building and training |
| **Optuna** | Bayesian hyperparameter optimisation (TPE sampler) |
| **scikit-learn** | ML models, CV, metrics, feature importance |
| **XGBoost** | Gradient boosted tree ensemble |
| **SHAP** | Model explainability for wallet ML models |
| **NetworkX** | Graph analysis for EDA |
| **Matplotlib / Seaborn** | Visualisation |
| **Google Colab / Kaggle** | GPU compute environment |
| **GitHub** | Version control and project publishing |

---

## ⚙️ DAX-style: GNN Design Formulas

### Class Weights

```python
w_illicit = n_total / (2 × n_illicit)   # ~5.1x — upweights the minority
w_licit   = n_total / (2 × n_licit)     # ~0.55x — downweights the majority
criterion = CrossEntropyLoss(weight=[w_licit, w_illicit])
```

### Xavier Initialisation

```
W ~ Uniform( -√(6 / (fan_in + fan_out)),  +√(6 / (fan_in + fan_out)) )
```
Maintains activation variance across layers — prevents vanishing/exploding gradients in multi-hop message passing.

### GraphNorm

```
x_norm = (x − α · μ_graph) / σ_graph
```
Where α is a **learned parameter** — the model decides how much of the graph-level mean to subtract. Applied after ReLU activation.

---

## 🚀 How to Use This Project

### Prerequisites

- Google Account (for Colab) — no local setup needed for notebooks
- Python 3.10+ (for local run)
- GPU recommended for GNN training

### Quickstart — Google Colab

Click any Colab badge in the Notebooks table above. The notebooks auto-download the dataset via `kagglehub`.

### Quickstart — Kaggle

1. Verify your phone number on Kaggle (required for GPU access)
2. Set **Settings → Accelerator → GPU T4 x2**
3. In the first cell, install only what Kaggle doesn't pre-install:

```python
import subprocess
subprocess.run(['pip', 'install', '-q', 'torch-geometric', 'optuna'], check=True)
# Do NOT reinstall torch — use Kaggle's pre-installed version
```

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/AML-GNN.git
cd AML-GNN
pip install -r requirements.txt
jupyter notebook
```

### Reproducing Full Results

Set `N_TRIALS = 100` in the experiment config cell for full paper-quality hyperparameter search. Notebooks default to 30 trials for speed.

---

## 📈 Conclusions

1. **GNNs effectively capture transaction-level fraud patterns** — graph topology provides signal that node features alone cannot deliver. Illicit transactions cluster in the network and exhibit unusual neighbourhood statistics.

2. **GATv2 performs best among all GNN architectures** — dynamic attention produces the highest AUPRC (0.534) with stable training. Val loss tracks train loss with no divergence across epochs.

3. **Temporal evaluation is non-negotiable** — random splits inflate results by leaking future fraud patterns into training. The temporal split reflects real deployment where future fraud is genuinely unknown.

4. **n=30 Optuna trials significantly outperforms n=10** — wider search space exploration finds more robust configurations rather than lucky ones.

5. **Wallet-level ML complements GNN transaction detection** — Random Forest on Elliptic++ wallet features achieves MCC=0.856 at the actor level, providing a second independent detection layer.

6. **Xavier initialisation consistently helps** — particularly for GATv2. GraphNorm benefits are architecture-dependent and do not uniformly improve performance.

---

## ⚠️ Limitations & Future Work

| Current Limitation | Future Direction |
|---|---|
| 77% unlabeled nodes limits supervised signal | Semi-supervised learning on unknown nodes |
| Computational constraints capped trials at 30 | Scale to 100+ trials with better hardware |
| Full-graph training is memory-intensive | Mini-batch via GraphSAINT / ClusterGCN |
| Severe class imbalance affects all models | GraphSMOTE for graph-level oversampling |
| Static graph — no edge-level timestamps | Temporal GNNs (TGN, TGAT) |
| Single dataset evaluation | Extend to Elliptic2 subgraph-level dataset |
| No real-time deployment | Build streaming AML inference pipeline |

---

## 📄 Full Report

The complete academic report is available in the `reports/` folder:  
📋 [`Final_Report.docx`](reports/Final_Report.docx)

The report covers: Introduction · Literature Review · Dataset Overview · Graph Construction · Methodology · Ablation Study · Results · Wallet ML Pipeline · Feature Importance · SHAP Analysis · Conclusions · Limitations & Future Work · References.

---

## 🔗 References

1. Dang, S.D., Nguyen, D.C., Dev, K., & Nijsse, J. (2026). *Normalisation and Initialisation Strategies for GNNs in Blockchain Anomaly Detection.* arXiv:2602.23599.
2. Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics.* KDD Workshop.
3. Kipf, T.N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
4. Veličković, P., et al. (2018). *Graph Attention Networks.* ICLR 2018.
5. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017.
6. Cai, T., et al. (2021). *GraphNorm: A Principled Approach to Accelerating GNN Training.* ICML 2021.
7. Glorot, X., & Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Networks.* AISTATS 2010.
8. Elmougy, Y., & Liu, L. (2023). *Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network.* KDD '23.
9. Elliptic Dataset — [kaggle.com/ellipticco/elliptic-data-set](https://www.kaggle.com/ellipticco/elliptic-data-set)

---

## 📜 Licence

This project is submitted as academic coursework for the MSc Statistics and Data Science programme at NMIMS Mumbai.  
The datasets are publicly available under their original licences at Kaggle.  
All model implementations, experimental design, and analysis are original work by the authors listed above.

---

*Built with ❤️ using PyTorch Geometric · Optuna · scikit-learn · April 2026*
