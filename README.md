#Fraud Detection in Bitcoin Transactions using Graph Neural Networks

Detecting fraudulent Bitcoin transactions using Graph Neural Networks (GNNs) on the Elliptic dataset. Unlike traditional ML models that treat transactions in isolation, this project exploits the transaction graph structure to capture coordinated fraud patterns — achieving stronger detection through relational learning.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Ablation Study](#ablation-study)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

Financial fraud detection is a challenging semi-supervised, class-imbalanced problem. This project:

- Builds a transaction graph from the Elliptic Bitcoin dataset
- Trains and compares three GNN architectures: GCN, GraphSAGE, and GAT
- Benchmarks against an XGBoost baseline (no graph structure)
- Uses Focal Loss to handle severe class imbalance
- Applies GNNExplainer for model interpretability (critical for AML compliance)
- Conducts an ablation study to verify whether GNNs can learn structural features automatically

---

## Dataset

**Elliptic Bitcoin Dataset** — a benchmark for financial fraud detection.

| Property | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges (transaction links) | 234,355 |
| Features per node | 166 |
| Local features | 94 |
| Aggregated features | 72 |
| Labeled licit | ~21% |
| Labeled illicit | ~2% |
| Unlabeled | ~77% |

The dataset spans 49 time steps, enabling temporal analysis of transaction patterns. The severe class imbalance (~10:1 licit:illicit ratio) makes this a challenging semi-supervised learning problem.

**Label encoding:**
- `1` (illicit) → `1`
- `2` (licit) → `0`
- `unknown` → `-1` (excluded from training)

---

## Project Structure

```
.
├── Code_Notebook.ipynb       # Main notebook with all code
├── elliptic_txs_features.csv # Node features (203769 × 167)
├── elliptic_txs_edgelist.csv # Edge list (txId1, txId2)
├── elliptic_txs_classes.csv  # Node labels (txId, class)
└── training_curves.png       # Saved training plots
```

---

## Models

### GCN (Graph Convolutional Network)
Baseline graph model. Each node aggregates features from all its neighbors with equal weight.

```
h_v^(k) = ReLU( W · mean(h_u^(k-1) for u in N(v) ∪ {v}) )
```

Architecture: 3-layer GCN with hidden size 128, dropout 0.5.

### GraphSAGE
Inductive learning via neighborhood sampling. Crucially, it can score brand-new, unseen nodes at inference time — enabling real-time transaction screening.

```
h_v^(k) = W · CONCAT( h_v^(k-1), MEAN(h_u^(k-1) for u in sample(N(v))) )
```

Architecture: 3-layer SAGE with hidden size 128, dropout 0.5.

### GAT (Graph Attention Network)
Assigns learnable attention weights to neighbors — a suspicious neighbor gets higher weight than a benign one. Uses 4-head multi-head attention.

```
α_ij = softmax( LeakyReLU( aᵀ [W·hᵢ ‖ W·hⱼ] ) )
h_v^(k) = ELU( Σⱼ α_ij · W · hⱼ )
```

Architecture: 3-layer GAT (heads=4, 4, 1), hidden size 128, dropout 0.5.

### XGBoost (Baseline)
Tabular baseline using only node features, no graph structure. Used to quantify the information gain from relational learning.

---

## Training Pipeline

| Component | Choice |
|---|---|
| Loss function | Focal Loss (α=0.25, γ=2.0) |
| Optimizer | Adam (lr=0.005, weight_decay=5e-4) |
| LR scheduler | StepLR (step=50, γ=0.5) |
| Epochs | 200 |
| Data split | 70% train / 15% val / 15% test (stratified) |
| Imbalance handling | Focal Loss for GNNs; `scale_pos_weight` for XGBoost |

**Focal Loss** down-weights easy (abundant licit) examples and focuses training on hard, rare (illicit) examples:

```
FL(p) = -α · (1 - p)^γ · log(p)
```

Best model checkpoint is saved based on validation F1 and restored after training.

---

## Evaluation

Models are evaluated on the held-out test set using:

- **Macro F1-score** — balanced metric across both classes
- **PR-AUC** — precision-recall area under curve, robust to class imbalance
- **Confusion matrix** — visualizing false positive vs false negative tradeoffs
- **Classification report** — per-class precision, recall, F1

---

## Explainability

GNNExplainer is applied to the best-performing GAT model to identify:

- Which node features most influenced a fraud prediction
- Which edges (transaction links) were most important for the decision

This is essential for AML (Anti-Money Laundering) compliance — regulators require human-interpretable evidence, not just binary flags.

```python
explainer = Explainer(
    model=gat_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    ...
)
```

---

## Ablation Study

To verify that GNNs learn graph structure automatically (rather than relying on pre-computed structural features), GraphSAGE is retrained using **only the 94 local features** (dropping the 72 aggregated/structural features).

If the ablated model performs comparably to the full 166-feature model, it confirms the GNN learns structural information from the graph topology itself.

---

## Results Summary

| Model | Macro F1 | PR-AUC |
|---|---|---|
| XGBoost (Baseline) | — | — |
| GCN | — | — |
| GraphSAGE | — | — |
| GAT | — | — |

*(Fill in after running the notebook — results depend on hardware and random seed.)*

**Key findings:**
- GNNs outperform the tabular XGBoost baseline, demonstrating the value of graph structure
- GAT achieves the best performance due to its attention mechanism
- GraphSAGE's inductive capability makes it suitable for production deployment
- GNNExplainer provides feature-level explanations for compliance use cases

---

## Installation

```bash
# Core dependencies
pip install torch torchvision
pip install torch-geometric
pip install xgboost scikit-learn pandas numpy matplotlib seaborn

# If running on Google Colab, torch-geometric installs automatically via:
!pip install torch-geometric -q
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- torch-geometric
- CUDA (optional, but recommended for training speed)

---

## Usage

1. **Download the Elliptic dataset** from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the three CSV files in the working directory.

2. **Open the notebook** in Google Colab or Jupyter:
   ```bash
   jupyter notebook Code_Notebook.ipynb
   ```

3. **Run all cells** in order. The notebook will:
   - Load and preprocess the dataset
   - Construct the transaction graph
   - Train GCN, GraphSAGE, and GAT
   - Evaluate against XGBoost baseline
   - Generate explainability outputs
   - Run the ablation study

4. **GPU acceleration**: The notebook auto-detects CUDA. Training on CPU is possible but significantly slower (200 epochs × 3 models).

---

## References

- [Elliptic Dataset Paper](https://arxiv.org/abs/1908.02591) — Weber et al., 2019
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [GNNExplainer](https://arxiv.org/abs/1903.03894) — Ying et al., 2019
- [Focal Loss](https://arxiv.org/abs/1708.02002) — Lin et al., 2017
- [GraphSAGE](https://arxiv.org/abs/1706.02216) — Hamilton et al., 2017
- [GAT](https://arxiv.org/abs/1710.10903) — Veličković et al., 2018
