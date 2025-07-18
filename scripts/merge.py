# scripts/merge.py

import openai
import pandas as pd
from datetime import datetime

def generate_merge_code(prompt: str, model: str = "gpt-4", temperature: float = 0.7):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful data scientist. Generate clean, correct pandas code to merge the following dataframes, "
                    "without unnecessary comments or explanations. Assume necessary data is already loaded into DataFrames."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"]

def execute_merge_code(code: str, context: dict):
    local_vars = dict(context)
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        raise RuntimeError(f"Error executing merge code: {e}")
    
    if "merged_df" not in local_vars:
        raise ValueError("Code must define a variable named 'merged_df'")
    return local_vars["merged_df"]

def save_merged_data(merged_df: pd.DataFrame, save_path: str = "data/merged.csv"):
    merged_df.to_csv(save_path, index=False)
    print(f"[{datetime.now()}] ✅ Merged data saved to {save_path}")
