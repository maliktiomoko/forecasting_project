# scripts/evaluate.py

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pandas as pd
import os

def load_target_scaler(path="data/target_scaler.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def inverse_transform_predictions(y_pred, scaler_path="data/target_scaler.pkl"):
    scaler = load_target_scaler(scaler_path)
    return scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

def evaluate_predictions(y_true, y_pred, inverse=False, scaler_path="data/target_scaler.pkl"):
    if inverse:
        y_true = inverse_transform_predictions(np.array(y_true), scaler_path)
        y_pred = inverse_transform_predictions(np.array(y_pred), scaler_path)
    
    mse = mean_squared_error(y_true, y_pred)
    print(f"[{datetime.now()}] 📊 MSE: {mse:.4f}")
    return mse

def save_submission(ids, y_pred, output_path="submissions/submission.csv", inverse=False, scaler_path="data/target_scaler.pkl"):
    if inverse:
        y_pred = inverse_transform_predictions(np.array(y_pred), scaler_path)

    df = pd.DataFrame({"id": ids, "target": y_pred})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[{datetime.now()}] 📤 Submission saved to {output_path}")
