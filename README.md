<div align="center">

# Credit Card Fraud Detection Model
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Oversampling-8A2BE2?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Multilayer%20Perceptron%20%28MLP%29%20%7C%20AUC%2097%25-228B22?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-blue?style=for-the-badge)
</div>

> Credit card fraud causes billions in losses for the finance industry each year. This project explores how machine learning can help detect fraudulent transactions automatically, even when fraud cases make up less than ~0.2% of all transactions. The core challenge is not building a model, but making it work reliably when the data is extremely imbalanced. This project tests multiple approaches to handle this imbalance, compares what works and what does not, and ends up with a **neural network trained on oversampled data** that achieves **97% AUC** and catches 83 out of 98 real fraud cases on completely unseen data.


## Project Overview

This project uses real credit card transaction data collected over two days in Europe to develop a fraud detection system. Transactions flagged as high risk can be blocked or sent for additional verification before processing. Neural Network trained with SMOTE oversampling achieved a highest **ROC-AUC score of 97%** compared to other models on the test set, successfully identifying **83 out of 98 fraudulent transactions** while minimizing false alerts.

## Dataset

| Property | Detail |
|---|---|
| Source | European credit card transactions |
| Period | Two consecutive days |
| Total records | 284,807 transactions |
| Features | 30 input features (V1–V28 via PCA, plus Time and Amount) |
| Target | `Class` — 0 = legitimate, 1 = fraud |
| Fraud cases | 492 (0.17%) |
| Legitimate cases | 284,315 (99.83%) |

> **Note:** The dataset is severely imbalanced (~99.8% vs ~0.2%), which is the central challenge addressed throughout this project. Features V1–V28 have already been transformed via PCA for confidentiality. Only `Time` and `Amount` required additional scaling.


## Project Structure

```
credit-card-fraud-detection/
│
├── creditcard.csv.gz          ← compressed dataset
├── notebook.ipynb             ← main analysis notebook
└── README.md                  ← this file
```

## Pipeline Steps

### Step 1: Data Profiling
- Initial shape and data type inspection
- Missing value check (none found)
- Class distribution analysis (confirmed severe imbalance - 99.8% vs 0.2%)
- Descriptive statistics review

### Step 2: Data Preprocessing
- Applied **RobustScaler** to `Time` and `Amount` columns (robust to outliers via median-based scaling)
- Left V1–V28 unchanged as they were already PCA-transformed and scaled
- Created balanced subsample for EDA: 394 fraud + 394 non-fraud transactions

### Step 3: Exploratory Data Analysis (EDA)
- Correlation heatmap to identify features with strongest relationship to fraud
  - Positive correlation: V2, V4, V11, V19, V20, V21
  - Negative correlation: V1, V3, V5, V6, V7, V9, V10, V12–V18
- Box plots and distribution plots for top correlated features
- Outlier removal using IQR method on top 5 correlated features (V14, V4, V12, V11, V10)
- Dimensionality reduction visualization using **t-SNE** and **PCA**
  - t-SNE achieved clear visual separation between fraud and non-fraud clusters
  - PCA showed partial but less distinct separation

### Step 4: Model Development and Evaluation

#### Classification Models on Undersampled Data
Four baseline classifiers were evaluated using 5-fold cross-validation:

| Model | Accuracy (CV) |
|---|---|
| Logistic Regression | 95% |
| K-Nearest Neighbors | 94% |
| Support Vector Classifier | 93% |
| Decision Tree | 89% |

GridSearchCV hyperparameter tuning was applied to all four models.

#### Undersampling Comparison — Random vs NearMiss

| Method | Accuracy | AUC | Precision | Recall |
|---|---|---|---|---|
| Random Undersampling (before split) | 94.3% | 93.1% | 2.7% | 91.8% |
| NearMiss (during training) | 50.8% | 71.7% | 0.3% | 92.7% |

> Random undersampling before the train-test split caused **data leakage** — the model appeared to perform well but failed on real imbalanced data. NearMiss applied correctly during training avoided leakage but showed **generalization failure**, misclassifying most legitimate transactions as fraud.

#### SMOTE Oversampling on Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | 97.5% |
| AUC | 94.7% |
| Precision | 6.0% |
| Recall | 91.8% |

Precision-Recall curve analysis showed that adjusting the decision threshold to **19.04** would achieve 0.83 precision and 0.82 recall simultaneously.

#### Neural Network (Undersampling vs SMOTE)

A 3-layer Neural Network (input → 32 → 2) was trained under both conditions:

| Condition | Architecture | Training Time |
|---|---|---|
| Undersampling | Dense(30) → Dense(32) → Dense(2, softmax) | 1.5s |
| SMOTE Oversampling | Dense(30) → Dense(32) → Dense(2, softmax) | 142.3s |

**Neural Network with SMOTE was the best performing model overall.**

## Final Results

| Model | AUC | Fraud Caught | False Positives |
|---|---|---|---|
| Neural Network + SMOTE | **97%** | **83 / 98** | 64 |
| Logistic Regression + SMOTE | 94.7% | ~90 / 98 | High |
| Random Undersampling (LR) | 93.1% | ~90 / 98 | Very High |

### Confusion Matrix Breakdown (Neural Network + SMOTE)

```
                    Predicted Legitimate    Predicted Fraud
Actual Legitimate        56,800+                  64          ← False Positives
Actual Fraud                15                    83          ← True Positives
```

- **True Positives (83):** Fraud cases correctly blocked
- **False Negatives (15):** Fraud cases that bypassed detection
- **False Positives (64):** Legitimate transactions incorrectly flagged

## Business Impact

| Outcome | Implication |
|---|---|
| 83 fraud cases caught | Direct reduction in chargeback costs and financial liability |
| 64 false positives out of ~57,000 | Minimal customer friction — 0.1% of legitimate transactions affected |
| 15 missed fraud cases | Unavoidable trade-off; represents ~15% miss rate on fraud |
| Low false positive rate | Fraud investigation team focuses on high-probability cases only |

## Key Findings

1. **Data leakage is the biggest risk**: undersampling before the train-test split inflates performance metrics and produces models that fail on real data
2. **SMOTE outperforms undersampling**: generating synthetic minority samples preserves information better than discarding majority samples
3. **Neural Networks generalize better** than logistic regression on this task when combined with SMOTE
4. **t-SNE confirms separability**: the features contain genuine discriminative signal between fraud and non-fraud
5. **Recall is more important than precision** in fraud detection, missing a fraud case is more costly than a false alarm

## Libraries Used

```
pandas, numpy          — data manipulation
matplotlib, seaborn    — visualization
scikit-learn           — preprocessing, models, evaluation
imbalanced-learn       — SMOTE, NearMiss, imbalanced pipeline
keras / tensorflow     — neural network
scipy                  — statistical distributions
```

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow keras scipy

# 2. Place dataset in project root
# creditcard.csv.gz should be in the same directory as the notebook

# 3. Open and run the notebook
jupyter notebook notebook.ipynb
```

## Future Improvements

- Expand dataset to capture wider range of fraud patterns across geographies and time periods
- Hyperparameter tuning with deeper neural network architectures (more hidden layers, dropout regularization)
- Benchmark against gradient boosting models (XGBoost, LightGBM, CatBoost)
- Ensemble methods: stacking or blending multiple models for improved robustness
- Interactive threshold optimization dashboard for business stakeholders to tune the precision-recall trade-off
- Reinforcement learning approaches for real-time adaptation to evolving fraud patterns

## Results Summary

A Neural Network model trained using SMOTE oversampling achieved a **ROC-AUC score of 97%** on the test set. The model successfully identified **83 out of 98 fraudulent transactions** while maintaining a balanced trade-off between false positives and false negatives. This demonstrates strong discriminatory power in detecting fraudulent transactions despite severe class imbalance, highlighting the model's potential for supporting real-world fraud detection systems.