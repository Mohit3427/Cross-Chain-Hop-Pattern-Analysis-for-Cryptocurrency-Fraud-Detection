# Cross-Chain Hop Pattern Analysis for Cryptocurrency Fraud Detection

> Minor Project | Department of Computer Science & Engineering | 2025–26

---

## Project Overview

This project introduces a novel machine learning pipeline for cryptocurrency fraud detection
based on **cross-chain hop pattern features** — a feature dimension completely absent from
all reviewed published research papers.

**Key contributions:**
- 14 novel hop-pattern features engineered from transaction graph structure
- SMOTE oversampling to handle severe class imbalance (2.3% fraud)
- 4 ML models (XGBoost, Random Forest, MLP, Logistic Regression) compared across 3 feature sets
- SHAP explainability — per-transaction fraud reasoning
- Direct comparison with 3 published research papers (GCN 2025, Ensemble 2024, ETHIAD 2025)

---

## Project Structure

```
project/
├── day1_crosschain_fraud_detection.ipynb   ← Data loading, EDA, hop feature engineering
├── day2_feature_engineering.ipynb          ← Interaction features, SMOTE, train/test split
├── day3_model_training.ipynb               ← 4 models × 3 feature sets, paper comparison
├── day4_shap_explainability.ipynb          ← SHAP values, waterfall, beeswarm plots
├── README.md                               ← This file
└── outputs/
    ├── feature_matrix.csv
    ├── model_results.csv
    ├── paper_comparison_table.csv
    ├── models/                             ← Saved .pkl model files
    └── plot1 … plot16 PNG files
```

---

## Setup & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap networkx joblib
```

### Dataset

1. Go to: https://github.com/git-disl/EllipticPlusPlus
2. Download ZIP → extract → place `Transactions Dataset/` folder in project root
3. Update `DATA_DIR` in Day 1 notebook to point to the folder

**OR on Google Colab:**
```python
!git clone https://github.com/git-disl/EllipticPlusPlus.git
DATA_DIR = 'EllipticPlusPlus/Transactions Dataset/'
```

---

## How to Run

Run notebooks in order:

| Notebook | What it does | Output |
|---|---|---|
| `day1_...ipynb` | Load data, build graph, engineer 8 hop features, 4 EDA plots | `feature_matrix.csv`, plot1–4 |
| `day2_...ipynb` | Add 6 interaction features, SMOTE, scale, split | 8 CSV splits, plot5–7 |
| `day3_...ipynb` | Train 4 models × 3 feature sets, paper comparison | `model_results.csv`, plot8–11 |
| `day4_...ipynb` | SHAP values, waterfall, beeswarm, dashboard | plot12–16 |

---

## Dataset

**Elliptic++ (KDD 2023)**
- 203,769 Bitcoin transactions
- 234,355 directed edges
- 46,564 labeled (4,545 illicit, 42,019 licit)
- 49 time steps, 166 features

Citation:
> Elmougy, Y., & Liu, L. (2023). Demystifying Fraudulent Transactions and Illicit Nodes
> in the Bitcoin Network for Financial Forensics. KDD '23. ACM.

---

## The 14 Novel Hop Features

| Feature | Fraud Signal |
|---|---|
| hop_count | High = layering |
| out_degree | Fan-out = fund splitting |
| in_degree | High = aggregation/mixing |
| time_span | Low = rapid movement |
| avg_time_gap | Low = fast laundering |
| fan_out_ratio | High = splitting funds |
| is_rapid_hop | 1 = suspicious pattern |
| neighbourhood_illicit_ratio | High = guilt by association |
| hop_density | Rapid layering signal |
| degree_asymmetry | Net splitting signal |
| illicit_exposure_score | Scaled contamination |
| rapid_fan_out | Combined rapid + split |
| normalised_time_span | Time per hop |
| connectivity_score | Total connection volume |

---

## Results Summary

See `outputs/model_results.csv` for full results.
See `outputs/paper_comparison_table.csv` for comparison with published papers.

### Research Paper Comparison

| Paper | F1 % | Cross-chain | XAI | Imbalance |
|---|---|---|---|---|
| GCN (Nature, 2025) | 97.20 | No | No | No |
| Ensemble (ETASR, 2024) | 98.50 | No | No | No |
| ETHIAD (PubMed, 2025) | 99.51 | No | Partial | ADASYN |
| **Ours** | **See CSV** | **Yes** | **SHAP** | **SMOTE** |

---

## Key Libraries

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥1.3 | Data manipulation |
| scikit-learn | ≥1.0 | ML models, metrics |
| xgboost | ≥1.6 | XGBoost classifier |
| imbalanced-learn | ≥0.9 | SMOTE oversampling |
| shap | ≥0.41 | Explainability |
| networkx | ≥2.6 | Transaction graph |
| matplotlib/seaborn | latest | Visualization |

---

## References

1. Elmougy & Liu (2023). Elliptic++ Dataset. KDD '23.
2. Somasundaram (2025). GCN Fraud Detection. Nature Scientific Reports.
3. Taher et al. (2024). Ensemble Fraud Detection. ETASR.
4. ETHIAD (2025). XGBoost + ADASYN. PubMed.
5. Chawla et al. (2002). SMOTE. JAIR, 16, 321–357.
6. Lundberg & Lee (2017). SHAP. NeurIPS 2017.
7. Chen & Guestrin (2016). XGBoost. KDD '16.

---

*Project by: [Your Name] | Guide: Prof. [Guide Name] | 2025–26*
