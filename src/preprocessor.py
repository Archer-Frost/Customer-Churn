import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_and_encode(df):
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    
    cat_cols = [col for col in X.select_dtypes(include="object").columns if col != "customerID"]
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    
    return preprocessor, X, y