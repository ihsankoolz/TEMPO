"""
================================================================================
UTILITY: Explore Non-Crisis Datasets
================================================================================
Explores all non-crisis datasets to identify:
1. Available CSV files and their structure
2. Text columns for tweet content
3. Timestamp columns for temporal features
4. Tweet ID columns for Snowflake timestamp extraction

Purpose:
    - Discover all non-crisis datasets in the folder
    - Identify column names (varies between datasets)
    - Determine which datasets need timestamp extraction
    - Generate summary report

Input:
    - non_crisis_data/*/*.csv (all CSV files in subfolders)

Output:
    - Console output showing dataset structures
    - non_crisis_summary.csv (optional summary file)

Usage:
    python utils/explore_non_crisis.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os

print("="*70)
print("EXPLORING NON-CRISIS DATASETS")
print("="*70)

non_crisis_folder = 'non_crisis_data'

dataset_info = []

for root, dirs, files in os.walk(non_crisis_folder):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            dataset_name = os.path.basename(root)

            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name}")
            print(f"File: {file}")
            print('='*70)

            try:
                df = pd.read_csv(file_path, nrows=5, encoding='utf-8', low_memory=False)
                full_df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

                print(f"Total rows: {len(full_df):,}")
                print(f"Columns: {df.columns.tolist()}")

                text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower() or 'content' in col.lower()]
                print(f"\nText columns: {text_cols if text_cols else 'NONE FOUND'}")

                time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'created' in col.lower()]
                print(f"Timestamp columns: {time_cols if time_cols else 'NONE FOUND'}")

                id_cols = [col for col in df.columns if 'id' in col.lower() and 'user' not in col.lower()]
                print(f"ID columns: {id_cols if id_cols else 'None'}")

                print(f"\nFirst row sample:")
                print(df.iloc[0].to_dict())

                dataset_info.append({
                    'dataset': dataset_name,
                    'file': file,
                    'rows': len(full_df),
                    'has_text': bool(text_cols),
                    'has_timestamp': bool(time_cols),
                    'has_id': bool(id_cols),
                    'text_col': text_cols[0] if text_cols else None,
                    'time_col': time_cols[0] if time_cols else None,
                    'id_col': id_cols[0] if id_cols else None
                })

            except Exception as e:
                print(f"Error reading file: {e}")

# Summary table
print(f"\n{'='*70}")
print("SUMMARY - NON-CRISIS DATASETS")
print('='*70)

summary_df = pd.DataFrame(dataset_info)
if not summary_df.empty:
    print("\nDataset Overview:")
    print(summary_df[['dataset', 'rows', 'has_text', 'has_timestamp', 'has_id']].to_string(index=False))

    total_tweets = summary_df['rows'].sum()
    print(f"\nTotal non-crisis tweets available: {total_tweets:,}")

    need_extraction = summary_df[~summary_df['has_timestamp'] & summary_df['has_id']]
    if not need_extraction.empty:
        print(f"\nDatasets needing timestamp extraction from IDs:")
        for _, row in need_extraction.iterrows():
            print(f"  - {row['dataset']}")

    summary_df.to_csv('non_crisis_summary.csv', index=False)
    print(f"\nSummary saved to: non_crisis_summary.csv")

else:
    print("No datasets found!")
