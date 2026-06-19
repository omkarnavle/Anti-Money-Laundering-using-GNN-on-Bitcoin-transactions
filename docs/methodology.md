# Methodology Notes

## GNN Track — Key Decisions

### Why Temporal Split Instead of Random Split

The Elliptic dataset has 49 time steps (each ~2 weeks). A random split allows the model to see time step 45 during training and be tested on time step 3 — predicting the past from the future. This is data leakage. The temporal split enforces:

- **Train:** time steps 1–29 (first ~15 months)
- **Val:** time steps 30–39
- **Test:** time steps 40–49

This means the model must generalise to *future* fraud patterns it has never seen.

### Why AUPRC Over Accuracy

Among labeled nodes, licit = 90.2% and illicit = 9.8%. A model predicting "all licit" achieves 90.2% accuracy but catches zero fraudsters. AUPRC measures the area under the Precision-Recall curve — a random classifier gets AUPRC ≈ 9.8% (the positive rate). Our models achieving AUPRC > 0.50 represent a 5x improvement over random.

### Why Weighted CrossEntropyLoss

Without weighting, gradient descent minimises loss by learning to predict licit for everything. The class weights are computed from training labels only (no leakage):

```
w_illicit = n_total / (2 × n_illicit)   # ~5.1x upweight
w_licit   = n_total / (2 × n_licit)     # ~0.55x downweight
```

### Xavier Initialisation

Maintains variance of activations across layers during forward pass. Prevents vanishing/exploding gradients in multi-hop message passing. Formula:

```
W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))
```

Benefits GATv2 and GraphSAGE most — both have separate weight matrices for source and target nodes that need balanced initialisation.

### GraphNorm

Normalises features using graph-level statistics (mean, std computed across all nodes in the graph) rather than batch statistics. Applied *after* ReLU activation:

```
x_norm = (x - α·μ_graph) / σ_graph
```

Where α is a *learned* parameter — the model decides how much of the mean to subtract. This is what distinguishes GraphNorm from LayerNorm.

### Why Unknown Nodes Stay in the Graph

Unknown nodes (77% of all nodes) are excluded from train/val/test masks — the model never tries to predict their labels. But they remain in the graph so message passing can route information through them. Removing them would disconnect the graph and destroy neighbourhood context.

---

## ML Track — Key Decisions

### Stratified Split

Unlike GNN (temporal), the wallet dataset uses stratified random 60/20/20 split. The wallet dataset does not have the same temporal ordering constraints since wallet behaviour is aggregated over time.

### Three-Method Feature Importance

Features are selected only if they rank highly across all three methods:
1. **MDI** (Mean Decrease in Impurity) — tree-based, fast but biased toward high-cardinality features
2. **Permutation** — model-agnostic, shuffle feature and measure accuracy drop
3. **Drop-column** — most rigorous, retrain without each feature and measure drop

Agreement across all three = genuinely important feature, not an artefact of one method.

### MCC as Primary Metric for Wallets

Matthews Correlation Coefficient is preferred over F1 for severely imbalanced binary classification because it accounts for all four cells of the confusion matrix (TP, TN, FP, FN). A model predicting all licit achieves MCC ≈ 0, immediately exposing it as useless — unlike accuracy (98%+) which masks the failure.
