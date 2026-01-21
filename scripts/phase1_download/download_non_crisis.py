"""
================================================================================
PHASE 1: DOWNLOAD - Non-Crisis Datasets from Kaggle
================================================================================
Downloads 8 non-crisis high-emotion datasets from Kaggle. These are events
that generate high Twitter activity but are NOT crises.

Purpose:
    - Provides contrast data for crisis detection
    - Helps model distinguish crisis emotions from excitement/entertainment
    - Covers sports, entertainment, and politics events

Output:
    - non_crisis_data/[dataset_folder]/*.csv

Datasets Downloaded (8 total):
    1. FIFA World Cup 2022 (sports)
    2. FIFA World Cup 2018 (sports)
    3. Tokyo Olympics 2020 (sports)
    4. ICC T20 World Cup 2021 (sports)
    5. US Election 2020 (politics)
    6. Game of Thrones Season 8 (entertainment)
    7. Coachella Festival 2015 (entertainment)
    8. Music Artists/Concerts (entertainment)

Prerequisites:
    - Kaggle API configured (~/.kaggle/kaggle.json)
    - pip install kaggle

Usage:
    python scripts/phase1_download/download_non_crisis.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import os
import subprocess
import zipfile
import pandas as pd

# Create directory for non-crisis data
os.makedirs('non_crisis_data', exist_ok=True)

print("="*70)
print("DOWNLOADING ALL NON-CRISIS DATASETS FROM KAGGLE")
print("="*70)

# All 8 non-crisis datasets
datasets = {
    'FIFA World Cup 2022': 'kumari2000/fifa-world-cup-twitter-dataset-2022',
    'FIFA World Cup 2018': 'rgupta09/world-cup-2018-tweets',
    'US Election 2020': 'manchunhui/us-election-2020-tweets',
    'Game of Thrones': 'monogenea/game-of-thrones-twitter',
    'Tokyo Olympics 2020': 'gpreda/tokyo-olympics-2020-tweets',
    'Coachella Festival': 'thedevastator/twitter-sentiment-analysis-coachella-festival',
    'Music Artists': 'alejandroservin/tweets-from-music-artists-balanced-dataset',
    'ICC T20 World Cup 2021': 'kaushiksuresh147/icc-t20-world-cup-2021-tweets'
}

downloaded = []
failed = []

# Save current directory
original_dir = os.getcwd()

for name, dataset_id in datasets.items():
    print(f"\n{'='*70}")
    print(f"Downloading: {name}")
    print(f"Dataset: {dataset_id}")
    print('='*70)

    try:
        # Create subfolder for this dataset
        folder_name = name.lower().replace(' ', '_')
        folder_path = os.path.join('non_crisis_data', folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Change to dataset folder
        os.chdir(folder_path)

        # Download using Kaggle API
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_id],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"Downloaded successfully!")

            # Find and extract ZIP file
            zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
            if zip_files:
                latest_zip = max(zip_files, key=os.path.getctime)
                print(f"Extracting: {latest_zip}")

                with zipfile.ZipFile(latest_zip, 'r') as zip_ref:
                    zip_ref.extractall('.')

                # Remove zip file
                os.remove(latest_zip)

                # List extracted files
                files = os.listdir('.')
                print(f"Extracted {len(files)} files:")
                for f in files[:5]:  # Show first 5
                    print(f"  - {f}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")

                downloaded.append((name, folder_name))

        else:
            print(f"Failed: {result.stderr}")
            failed.append(name)

        # Go back to original directory
        os.chdir(original_dir)

    except Exception as e:
        print(f"Error: {e}")
        failed.append(name)
        os.chdir(original_dir)

# Summary
print(f"\n{'='*70}")
print("DOWNLOAD SUMMARY")
print('='*70)
print(f"Successfully downloaded: {len(downloaded)}/{len(datasets)}")
for name, folder in downloaded:
    print(f"  - {name} -> non_crisis_data/{folder}/")

if failed:
    print(f"\nFailed: {len(failed)}")
    for name in failed:
        print(f"  - {name}")

print(f"\n{'='*70}")
print("Next step: Run phase3_standardize/standardize_non_crisis_data.py")
print('='*70)
