import numpy as np
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from feature_engineering import engineer_features
from preprocessor import preprocess_and_encode

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"
    TEST_PATH = BASE_DIR / "data" / "test.csv"
    OUT_PATH = BASE_DIR / "prediction" / "test_predictions.csv"

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")

    if not TEST_PATH.exists():
        raise FileNotFoundError("Test file not found.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    df = engineer_features(df)
    preprocessor,X,y = preprocess_and_encode(df)
    CIDs = X["customerID"]
    X = X.drop(columns=["customerID"])

    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]

    feature_names = [f.split("__",1)[-1] for f in pre.get_feature_names_out()]

    coefs = clf.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    })

    importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

    print("\n=== TOP FEATURES (by absolute coefficient) ===")
    print(importance_df.head(15))
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # save predictions
    output = pd.DataFrame({
        "Customer ID": CIDs,
        "actual_churn": y,
        "predicted_churn": y_pred,
        "churn_probability": y_prob
    })

    # print metrics
    print("=== TEST PERFORMANCE ===")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("F1 Score :", f1_score(y, y_pred))
    print("ROC-AUC  :", roc_auc_score(y, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    (BASE_DIR / "prediction").mkdir(exist_ok=True)
    output.to_csv(OUT_PATH, index=False)

    
    print(f"\nSaved predictions to: {OUT_PATH}")

if __name__ == "__main__":
    main()

