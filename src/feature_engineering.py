import pandas as pd

def engineer_features(df):
    df["tenure_group"] = pd.cut(df["tenure"], bins=[0,12,24,48,72]).astype(str)

    services = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
        ]

    df["num_services"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

    df["high_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    df["contract_length"] = df["Contract"].map({
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
        })
    
    return df