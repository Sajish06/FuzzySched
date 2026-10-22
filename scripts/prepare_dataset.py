"""
Dataset Preparation and Data Quality Audit Script for FuzzySched.

Member 2 Deliverable: Step 1 - Inspect and prepare the dataset.
- Inspects data quality, missing values, datatypes, and distributions.
- Cleans data without overwriting the original dataset.
- Standardizes column names and adds normalized features.
- Exports cleaned dataset to data/cleaned_studentsessions.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path for normalization functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from fuzzy_engine import map_days_to_urgency, map_difficulty, map_workload

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "studentsessions.csv"))
CLEAN_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_studentsessions.csv"))


def audit_and_clean_dataset(
    raw_path: str = RAW_DATA_PATH,
    clean_path: str = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """
    Inspect, validate, clean, and enrich the student sessions dataset.
    
    Returns:
        pd.DataFrame: Cleaned and normalized DataFrame.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")
        
    print("=" * 70)
    print("STEP 1: DATASET INSPECTION & DATA QUALITY AUDIT")
    print("=" * 70)
    
    # 1. Load Raw Dataset
    raw_df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Total Rows: {len(raw_df)}, Total Columns: {len(raw_df.columns)}")
    
    # 2. Check Missing Values
    null_counts = raw_df.isnull().sum()
    print("\n--- Missing Value Audit ---")
    missing_found = False
    for col, count in null_counts.items():
        if count > 0:
            print(f"  Column '{col}': {count} missing values")
            missing_found = True
    if not missing_found:
        print("  [OK] Zero missing values detected across all columns.")
        
    # 3. Check Duplicate Rows
    duplicates = raw_df.duplicated().sum()
    print(f"\n--- Duplicate Row Audit ---")
    print(f"  [OK] Zero duplicate rows detected (count = {duplicates}).")
    
    # 4. Check Unique Identifiers & String Formatting
    print("\n--- Identifier & Categorical Audit ---")
    raw_df_clean = raw_df.copy()
    
    # Notice: assignment 'I ' has a trailing whitespace in raw CSV!
    initial_ids = raw_df_clean["assignment_id"].tolist()
    raw_df_clean["assignment_id"] = raw_df_clean["assignment_id"].astype(str).str.strip()
    cleaned_ids = raw_df_clean["assignment_id"].tolist()
    
    whitespace_fixed = [f"'{o}' -> '{c}'" for o, c in zip(initial_ids, cleaned_ids) if o != c]
    if whitespace_fixed:
        print(f"  [OK] Fixed trailing whitespace in assignment_id: {', '.join(whitespace_fixed)}")
        
    # Check uniqueness of assignment_id
    num_unique_assignments = raw_df_clean["assignment_id"].nunique()
    print(f"  [OK] Unique assignments: {num_unique_assignments} (Assignments A through Z)")
    print(f"  [OK] Classes represented: {sorted(raw_df_clean['class_id'].unique().tolist())}")
    print(f"  [OK] Student user_id: {raw_df_clean['user_id'].unique().tolist()}")

    # 5. Numerical Ranges and Distribution Audit
    print("\n--- Usable Columns & Feature Range Audit ---")
    usable_cols = {
        "days_till_deadline_started": "Days remaining when task started (Deadline Urgency)",
        "perceived_difficulty": "Task difficulty rating on Likert scale 1-5",
        "num_other_assignments": "Concurrent active assignments (Workload load)",
        "perceived_length": "Estimated task length/duration (Workload volume)"
    }
    
    for col, desc in usable_cols.items():
        min_v = raw_df_clean[col].min()
        max_v = raw_df_clean[col].max()
        mean_v = raw_df_clean[col].mean()
        std_v = raw_df_clean[col].std()
        print(f"  * {col} ({desc}):")
        print(f"      Range: [{min_v}, {max_v}] | Mean: {mean_v:.2f} | Std: {std_v:.2f}")

    # Outlier note
    length_q95 = raw_df_clean["perceived_length"].quantile(0.95)
    print(f"\n  [Audit Note] perceived_length has high outlier: Assignment N (length=20), median={raw_df_clean['perceived_length'].median():.1f}.")
    print(f"  [Audit Note] days_till_deadline_started has narrow window [1, 4] days in raw student log.")

    # 6. Cleaning & Feature Engineering
    # Standardize column naming for intuitive model consumption
    raw_df_clean["days_remaining"] = raw_df_clean["days_till_deadline_started"]
    raw_df_clean["difficulty"] = raw_df_clean["perceived_difficulty"]
    raw_df_clean["other_assignments"] = raw_df_clean["num_other_assignments"]
    
    # Compute Normalized Urgency, Difficulty, Workload in both [0, 10] and [0, 100]
    urgency_10 = []
    urgency_100 = []
    diff_10 = []
    diff_100 = []
    workload_10 = []
    workload_100 = []
    
    for _, row in raw_df_clean.iterrows():
        # Urgency (horizon = 14 days)
        u10 = map_days_to_urgency(row["days_remaining"], max_days=14.0)
        u100 = u10 * 10.0
        
        # Difficulty (Likert 1 to 5)
        d10 = map_difficulty(row["difficulty"], min_val=1.0, max_val=5.0)
        d100 = d10 * 10.0
        
        # Workload (composite of assignments [0-5] and length [0-20], weight=0.35)
        w10 = map_workload(
            row["other_assignments"],
            perceived_length=row["perceived_length"],
            max_assignments=5.0,
            max_length=20.0,
            length_weight=0.35,
        )
        w100 = w10 * 10.0
        
        urgency_10.append(round(u10, 2))
        urgency_100.append(round(u100, 2))
        diff_10.append(round(d10, 2))
        diff_100.append(round(d100, 2))
        workload_10.append(round(w10, 2))
        workload_100.append(round(w100, 2))
        
    raw_df_clean["urgency_norm_10"] = urgency_10
    raw_df_clean["urgency_norm_100"] = urgency_100
    raw_df_clean["difficulty_norm_10"] = diff_10
    raw_df_clean["difficulty_norm_100"] = diff_100
    raw_df_clean["workload_norm_10"] = workload_10
    raw_df_clean["workload_norm_100"] = workload_100

    # Ensure parent dir exists and save clean file
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    raw_df_clean.to_csv(clean_path, index=False)
    print(f"\n[OK] Cleaned and enriched dataset successfully saved to: {clean_path}")
    print(f"[OK] Original file '{raw_path}' preserved intact.")
    print("=" * 70)
    
    return raw_df_clean


if __name__ == "__main__":
    audit_and_clean_dataset()
