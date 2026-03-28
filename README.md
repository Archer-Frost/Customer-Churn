# Customer-Churn

End-to-end machine learning project to predict customer churn using classification models, with a focus on balancing business objectives such as maximizing churn detection while controlling false positives.

## Overview

Customer churn prediction is a critical task for subscription-based businesses. This project builds and evaluates multiple models to identify customers likely to churn, with emphasis on:

- Handling class imbalance
- Interpreting model performance using appropriate metrics
- Selecting models aligned with business objectives

## Dataset

Telco Customer Churn dataset (Kaggle)
Target variable: Churn (Yes/No)
Features include:
- Customer demographics
- Account information
- Service usage patterns
- Billing and contract details

## Project Workflow

### 1. Data preparation
- Converted TotalCharges to numeric
- Handled missing values
- Stratified train-test split (80-20)

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis (tenure, charges)
- Categorical feature relationships with churn
- Identification of churn patterns across segments

### 3. Feature Engineering
- Tenure grouping
- Service count features
- Contract encoding
- Interaction features

### 4. Modeling

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Final Model Chosen : Logistic Regression

Pipeline:

- Preprocessing (encoding + scaling)
- Cross-validation (StratifiedKFold)
- Hyperparameter tuning (GridSearchCV)

### 5. Evaluation Strategy

Due to class imbalance:

- ROC-AUC → model selection (threshold-independent)
- Precision–Recall curve → trade-off analysis
- Recall (churn class) → primary business metric
- F1-score → balance metric

## Results

Logistic Regression (Final Model)
- ROC-AUC: ~0.84
- Recall (Churn): ~0.80
- F1 Score: ~0.61
- Achieved ~80% recall on churn class with ROC-AUC ~0.84 using Logistic Regression.

Key observations:

- High recall ensures most churners are identified
- Acceptable precision trade-off for business use
- Strong overall discrimination ability

### Model Selection

Although XGBoost achieved slightly higher ROC-AUC and accuracy, Logistic Regression was selected due to:

- Higher recall for the churn class
- Better balance (F1-score)
- Competitive performance across evaluation metrics
- Interpretability of feature effects

### Project Structure
```
Customer-Churn/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Modelling_baseline_with_FE.ipynb
│   ├── LogisticRegression.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocessor.py
│   ├── feature_engineering.py
│
├── models/
│   └── logistic_regression_model.joblib
│
├── prediction/
│   └── test_predictions.csv
│
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/Archer-Frost/Customer-Churn.git
cd Customer-Churn
```

Train model

```bash
python src/train.py
```

Generate predictions

```bash
python src/predict.py
```

## Key Takeaways

- Model selection should align with business objectives, not just accuracy
- Recall is critical in churn prediction
- Threshold-independent metrics (ROC-AUC) are essential for fair comparison
- Logistic Regression remains a strong baseline with high interpretability

## Author

Anupam Dasgupta

