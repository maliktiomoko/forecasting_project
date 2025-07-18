# scripts/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

def preprocess_data(df: pd.DataFrame, date_column: str = "date", save_scaler_path: str = "models/scaler.pkl"):
    print(f"[{datetime.now()}] 🔄 Starting preprocessing...")

    df = df.copy()

    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])

    # Feature engineering: date parts
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["weekday"] = df[date_column].dt.weekday
    df["weekofyear"] = df[date_column].dt.isocalendar().week

    # Drop original date column (optional)
    df.drop(columns=[date_column], inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Encode categoricals
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes

    # Scale numerical features (excluding target)
    scaler = StandardScaler()
    feature_cols = df.columns.difference(["target"])
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save the scaler
    os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
    with open(save_scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[{datetime.now()}] ✅ Preprocessing complete. Scaler saved to {save_scaler_path}")
    return df
