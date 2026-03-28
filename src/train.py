from pathlib import Path
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

from preprocessor import preprocess_and_encode
from feature_engineering import engineer_features

import random

random.seed(42)
np.random.seed(42)

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "train.csv"
    df = pd.read_csv(DATA_PATH)

    df = engineer_features(df)
    preprocessor,X,y = preprocess_and_encode(df)

    logistic_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=42
            ))
        ])
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = [
        {
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs"]
        },
        {
            "clf__penalty": ["l1"],
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear"]
        }
    ]

    grid = GridSearchCV(
        logistic_pipe,
        param_grid,
        cv=kf,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X, y)

    os.makedirs("models",exist_ok = True)
    joblib.dump(grid.best_estimator_,"models/logistic_regression_model.joblib")
    print("Saved: models/logistic_regression_model.joblib")

if __name__ == "__main__":
    main()