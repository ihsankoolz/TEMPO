"""
================================================================================
PHASE 1: DOWNLOAD - Baseline Noise Tweets (Sentiment140)
================================================================================
Downloads the Sentiment140 dataset from Kaggle as baseline/noise tweets.
This provides general Twitter content unrelated to any specific event.

Purpose:
    - Provides baseline noise for training
    - Helps model distinguish crisis from normal Twitter activity
    - 1.6 million general tweets

Output:
    - baseline_data/training.1600000.processed.noemoticon.csv

Prerequisites:
    - Kaggle API configured (~/.kaggle/kaggle.json)
    - pip install kaggle

Usage:
    python scripts/phase1_download/download_baseline.py

Note:
    The file has NO header row. Column order is:
    sentiment, id, date, query, user, text

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import subprocess
import zipfile
import os

print("="*70)
print("DOWNLOADING SENTIMENT140 - BASELINE TWEETS")
print("="*70)

os.makedirs('baseline_data', exist_ok=True)
os.chdir('baseline_data')

# Download
print("\nDownloading... (This is ~238 MB, might take a few minutes)")
result = subprocess.run(
    ['kaggle', 'datasets', 'download', '-d', 'kazanova/sentiment140'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("Downloaded!")

    # Extract
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    if zip_files:
        print(f"Extracting {zip_files[0]}...")
        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_files[0])

        files = os.listdir('.')
        print(f"\nExtracted files: {files}")

        # Check the CSV
        import pandas as pd
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            print(f"\nExamining {csv_files[0]}...")
            df = pd.read_csv(
                csv_files[0],
                encoding='latin-1',  # Sentiment140 uses latin-1
                nrows=10,
                header=None,  # No header row
                names=['sentiment', 'id', 'date', 'query', 'user', 'text']
            )

            print(f"\nColumns (inferred): {df.columns.tolist()}")
            print(f"\nFirst tweet:")
            print(df.iloc[0])

            # Count total rows
            print(f"\nCounting total rows...")
            full_count = sum(1 for _ in open(csv_files[0], encoding='latin-1'))
            print(f"Total baseline tweets: {full_count:,}")

else:
    print(f"Error: {result.stderr}")

os.chdir('..')  # Back to main folder
print("\nBaseline dataset ready!")
