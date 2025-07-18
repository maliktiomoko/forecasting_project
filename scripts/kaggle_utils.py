# scripts/kaggle_utils.py

import os
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime

def setup_kaggle(config_path="config/kaggle.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Kaggle API key not found at {config_path}")

    os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(config_path)
    api = KaggleApi()
    api.authenticate()
    print(f"[{datetime.now()}] ✅ Kaggle API authenticated.")
    return api

def submit_to_kaggle(api, file_path, competition_name, message="Auto submission"):
    print(f"[{datetime.now()}] 📤 Submitting {file_path} to {competition_name}...")
    api.competition_submit(file_path, message, competition_name)
    print(f"[{datetime.now()}] ✅ Submission complete.")

def get_latest_submission_score(api, competition_name):
    subs = api.competition_submissions(competition_name)
    if not subs:
        print(f"[{datetime.now()}] ⚠️ No submissions found.")
        return None

    latest = sorted(subs, key=lambda x: x.date, reverse=True)[0]
    score_data = {
        "submitted_at": latest.date,
        "public_score": latest.publicScore,
        "private_score": latest.privateScore,
        "status": latest.status,
        "description": latest.description,
    }

    print(f"[{datetime.now()}] 📊 Latest submission: {score_data}")
    return score_data
