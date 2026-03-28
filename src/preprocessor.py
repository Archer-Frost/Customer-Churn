import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_and_encode(df):
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # target
    target = "Churn"
    
    # separate features
    y = df["Churn"]
    X = df.drop(columns=[target])
    
    # identify columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    
    # transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    
    return preprocessor, X, y