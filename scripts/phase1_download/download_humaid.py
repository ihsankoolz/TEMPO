"""
================================================================================
PHASE 1: DOWNLOAD - HumAID Crisis Dataset
================================================================================
Downloads the HumAID dataset from CrisisNLP containing crisis tweets from
19 major disaster events (2016-2019).

Purpose:
    - Provides crisis tweets with event labels
    - 77K labeled tweets from hurricanes, earthquakes, floods, wildfires
    - Used for training crisis detection

Output:
    - crisis_datasets/humaid_crisis_data/ (multiple TSV files per event)

Events Included:
    - Hurricane Harvey, Irma, Florence, Matthew
    - California Wildfires, Canada Wildfires, Greece Wildfires
    - Nepal Earthquake, Mexico Earthquake, Kaikoura Earthquake
    - Kerala Floods, Sri Lanka Floods, Maryland Floods
    - And more...

Usage:
    python scripts/phase1_download/download_humaid.py

Note:
    This downloads Set 1 (47K tweets). For Set 2 (29K more),
    fill out the form at: https://crisisnlp.qcri.org/humaid_dataset

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import requests
import tarfile
import os

# Create directory
os.makedirs('crisis_datasets/humaid_crisis_data', exist_ok=True)

print("Downloading HumAID Set 1 (47K tweets)...")
print("This is ~76 MB, might take a few minutes...")

url = "https://crisisnlp.qcri.org/data/humaid/HumAID_data_events_set1_47K.tar.gz"

# Download
response = requests.get(url, stream=True)
file_path = 'crisis_datasets/humaid_crisis_data/HumAID_set1.tar.gz'

with open(file_path, 'wb') as f:
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        downloaded += len(chunk)
        if total > 0:
            percent = (downloaded / total) * 100
            print(f'\rProgress: {percent:.1f}%', end='')

print("\nDownload complete!")

# Extract
print("\nExtracting files...")
with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall('crisis_datasets/humaid_crisis_data/')
print("Extraction complete!")

# List what's inside
print("\nFiles extracted:")
for root, dirs, files in os.walk('crisis_datasets/humaid_crisis_data/'):
    for file in files:
        if file.endswith('.tsv'):
            file_path = os.path.join(root, file)
            print(f"  - {file}")

            # Quick peek at first file
            import pandas as pd
            try:
                df = pd.read_csv(file_path, sep='\t', nrows=5)
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Sample row: {df.iloc[0].to_dict()}")
                break
            except:
                pass

print("\nHumAID Set 1 ready in 'crisis_datasets/humaid_crisis_data/' folder!")
print("\nNext step: Run phase2_process/extract_humaid_timestamps.py to get timestamps")
