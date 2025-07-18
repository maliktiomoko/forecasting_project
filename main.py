# main.py

import pandas as pd
from datetime import datetime

# Import all steps
from scripts.merge import generate_merge_code, execute_merge_code, save_merged_data
from scripts.preprocess import preprocess_data
from scripts.split import split_data
from scripts.train import train_model, predict, save_model
from scripts.evaluate import evaluate_predictions, save_submission
from scripts.kaggle_utils import setup_kaggle, submit_to_kaggle, get_latest_submission_score

# --- Configuration ---
MERGE_PROMPT_PATH = "prompts/merge.txt"
MERGED_DATA_PATH = "data/merged.csv"
PROCESSED_DATA_PATH = "data/processed.csv"
SUBMISSION_PATH = "submissions/submission_M1_P1_S1.csv"
COMPETITION_NAME = "store-sales-time-series-forecasting"
TARGET_COL = "target"

def run_pipeline():
    print(f"\n🚀 Starting pipeline at {datetime.now()}\n")

    # --- 1. Load raw data ---
    print("📥 Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    stores_df = pd.read_csv("data/stores.csv")
    oil_df = pd.read_csv("data/oil.csv")
    holidays_df = pd.read_csv("data/holidays_events.csv")
    transactions_df = pd.read_csv("data/transactions.csv")

    context = {
        "train_df": train_df,
        "test_df": test_df,
        "stores_df": stores_df,
        "oil_df": oil_df,
        "holidays_df": holidays_df,
        "transactions_df": transactions_df,
    }

    # --- 2. Generate and execute merge code using GPT ---
    print("🧠 Generating merge logic using GPT...")
    with open(MERGE_PROMPT_PATH, "r") as f:
        prompt = f.read()

    merge_code = generate_merge_code(prompt)
    merged_df = execute_merge_code(merge_code, context)
    save_merged_data(merged_df, MERGED_DATA_PATH)

    # --- 3. Preprocess ---
    print("🧹 Preprocessing...")
    merged_df = pd.read_csv(MERGED_DATA_PATH)
    processed_df = preprocess_data(merged_df)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    # --- 4. Split ---
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_df, target_col=TARGET_COL, test_size=0.2)

    # --- 5. Train ---
    print("🏋️ Training model...")
    model = train_model(X_train, y_train)
    save_model(model)

    # --- 6. Predict & Evaluate ---
    print("🔮 Predicting and evaluating...")
    y_pred = predict(model, X_test)
    evaluate_predictions(y_test, y_pred, inverse=True)

    # --- 7. Save submission ---
    print("📝 Preparing submission file...")
    test_ids = X_test.index  # Or use a column if present
    save_submission(test_ids, y_pred, output_path=SUBMISSION_PATH, inverse=True)

    # --- 8. Submit to Kaggle ---
    print("🚀 Submitting to Kaggle...")
    api = setup_kaggle("config/kaggle.json")
    submit_to_kaggle(api, SUBMISSION_PATH, COMPETITION_NAME, message="Try M1_P1_S1")
    get_latest_submission_score(api, COMPETITION_NAME)

    print(f"\n✅ Pipeline completed at {datetime.now()}\n")

if __name__ == "__main__":
    run_pipeline()
