# scripts/split.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

def split_data(df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, scale_target: bool = True, save_dir: str = "data"):
    print(f"[{datetime.now()}] 📤 Splitting data...")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if scale_target:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Save target scaler
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)

        y_train = y_train_scaled
        y_test = y_test_scaled

        print(f"[{datetime.now()}] ✅ Target scaled and saved to {os.path.join(save_dir, 'target_scaler.pkl')}")
    else:
        print(f"[{datetime.now()}] ⚠️ Target NOT scaled.")

    return X_train, X_test, y_train, y_test
