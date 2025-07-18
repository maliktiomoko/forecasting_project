# scripts/train.py

import lightgbm as lgb
import pickle
import os
from datetime import datetime

def train_model(X_train, y_train, params=None):
    print(f"[{datetime.now()}] 🚀 Training LightGBM model...")

    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_estimators": 100,
    }

    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)

    print(f"[{datetime.now()}] ✅ Model trained.")
    return model

def predict(model, X_test):
    print(f"[{datetime.now()}] 📈 Generating predictions...")
    return model.predict(X_test)

def save_model(model, save_path="models/lightgbm_model.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[{datetime.now()}] 💾 Model saved to {save_path}")

def load_model(path="models/lightgbm_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
