"""
================================================================================
PHASE 3: STANDARDIZE - Non-Crisis Datasets
================================================================================
Standardizes all non-crisis datasets (sports, entertainment, politics) to a
common format for multi-task BERT training.

Purpose:
    - Unify column headers across 8 different non-crisis datasets
    - Extract only essential columns (text, timestamp)
    - Assign event_name and event_type labels
    - Combine into single non_crisis_combined.csv

Input:
    - non_crisis_data/[dataset_folders]/*.csv

Output:
    - standardized_data/[dataset]_standardized.csv (individual files)
    - standardized_data/non_crisis_combined.csv (combined file)

Standard Columns:
    text, created_at, event_name, event_type, crisis_label, source_dataset

Event Types:
    - sports: FIFA World Cup, Olympics, ICC T20
    - entertainment: Coachella, Game of Thrones, Music Concerts
    - politics: US Election 2020

Note:
    tweet_id is NOT included due to Excel precision issues with large numbers.
    This is fine for BERT training which only needs text and labels.

Usage:
    python scripts/phase3_standardize/standardize_non_crisis_data.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os
from datetime import datetime

print("="*80)
print("STANDARDIZING NON-CRISIS DATASETS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = "./non_crisis_data/"
OUTPUT_DIR = "./standardized_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

datasets_config = {
    'coachella': {
        'filename': 'coachella.csv',
        'text_col': 'text',
        'time_col': 'tweet_created',
        'event_name': 'coachella_2015',
        'event_type': 'entertainment',
        'encoding': 'utf-8'
    },
    'fifa_worldcup': {
        'filename': 'fifa_worldcup_2022.csv',
        'text_col': 'Tweet Content',
        'time_col': 'Tweet Posted Time',
        'event_name': 'fifa_worldcup_2022',
        'event_type': 'sports',
        'encoding': 'utf-8'
    },
    'music_concerts': {
        'filename': 'music_artists.csv',
        'text_col': 'text',
        'time_col': 'created_at',
        'event_name': 'music_concerts_2021',
        'event_type': 'entertainment',
        'encoding': 'utf-8'
    },
    'tokyo_olympics': {
        'filename': 'tokyo_olympics_2020.csv',
        'text_col': 'text',
        'time_col': 'date',
        'event_name': 'tokyo_olympics_2020',
        'event_type': 'sports',
        'encoding': 'utf-8'
    },
    'us_election': {
        'filename': 'us_election_2020.csv',
        'text_col': 'tweet',
        'time_col': 'created_at',
        'event_name': 'us_election_2020',
        'event_type': 'politics',
        'encoding': 'utf-8'
    },
    'game_of_thrones': {
        'filename': 'game_of_thrones.csv',
        'text_col': 'text',
        'time_col': 'created_at',
        'event_name': 'got_season8_2019',
        'event_type': 'entertainment',
        'encoding': 'utf-8'
    },
    'worldcup_2018': {
        'filename': 'FIFA.csv',
        'text_col': 'Tweet',
        'time_col': 'Date',
        'event_name': 'fifa_worldcup_2018',
        'event_type': 'sports',
        'encoding': 'utf-8'
    },
    'icc_t20': {
        'filename': 't20_tweets.csv',
        'text_col': 'text',
        'time_col': 'date',
        'event_name': 'icc_t20_worldcup_2021',
        'event_type': 'sports',
        'encoding': 'utf-8'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_timestamp(timestamp_str):
    """Convert various timestamp formats to standard datetime"""
    if pd.isna(timestamp_str):
        return None

    formats = [
        '%d/%m/%Y %H:%M',
        '%d %b %Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M'
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(timestamp_str, format=fmt)
        except:
            continue

    try:
        return pd.to_datetime(timestamp_str)
    except:
        return None

def standardize_dataset(config, dataset_name):
    """Standardize a single dataset to common format"""
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name.upper()}")
    print(f"{'='*80}")

    filepath = os.path.join(INPUT_DIR, config['filename'])

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, encoding=config['encoding'], low_memory=False)
        print(f"Loaded: {len(df):,} rows")
        print(f"Original columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return None

    # Extract required columns (tweet_id removed due to Excel precision issues)
    try:
        standardized = pd.DataFrame({
            'text': df[config['text_col']].astype(str),
            'created_at': df[config['time_col']],
            'event_name': config['event_name'],
            'event_type': config['event_type'],
            'crisis_label': 0,
            'source_dataset': dataset_name
        })
        print(f"Extracted columns: text, created_at")

    except KeyError as e:
        print(f"ERROR: Column not found: {e}")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Clean timestamps
    print(f"Cleaning timestamps...")
    standardized['created_at'] = standardized['created_at'].apply(clean_timestamp)

    # Remove rows with missing text or timestamp
    before_clean = len(standardized)
    standardized = standardized.dropna(subset=['text', 'created_at'])
    after_clean = len(standardized)

    if before_clean > after_clean:
        print(f"Removed {before_clean - after_clean:,} rows with missing text/timestamp")

    # Remove duplicates
    before_dedup = len(standardized)
    standardized = standardized.drop_duplicates(subset=['text', 'created_at'])
    after_dedup = len(standardized)

    if before_dedup > after_dedup:
        print(f"Removed {before_dedup - after_dedup:,} duplicate tweets")

    print(f"Final row count: {len(standardized):,}")
    print(f"Event: {config['event_name']} (Type: {config['event_type']})")

    # Save individual standardized file
    output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_standardized.csv")
    standardized.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return standardized

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print(f"\nInput Directory: {INPUT_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"\nDatasets to process: {len(datasets_config)}")

all_datasets = []

for dataset_name, config in datasets_config.items():
    result = standardize_dataset(config, dataset_name)
    if result is not None:
        all_datasets.append(result)
    else:
        print(f"Skipping {dataset_name} due to errors")

# ============================================================================
# COMBINE ALL DATASETS
# ============================================================================

if len(all_datasets) > 0:
    print(f"\n{'='*80}")
    print("COMBINING ALL NON-CRISIS DATASETS")
    print(f"{'='*80}")

    combined = pd.concat(all_datasets, ignore_index=True)

    print(f"\nCOMBINED DATASET SUMMARY:")
    print(f"   Total tweets: {len(combined):,}")
    print(f"\n   By Event Type:")
    print(combined['event_type'].value_counts().to_string())
    print(f"\n   By Event Name:")
    print(combined['event_name'].value_counts().to_string())
    print(f"\n   Date Range:")
    print(f"   Earliest: {combined['created_at'].min()}")
    print(f"   Latest: {combined['created_at'].max()}")

    # Save combined file
    combined_file = os.path.join(OUTPUT_DIR, "non_crisis_combined.csv")
    combined.to_csv(combined_file, index=False)
    print(f"\nCOMBINED FILE SAVED: {combined_file}")

    print(f"\n{'='*80}")
    print("STANDARDIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"   - Individual files: {len(all_datasets)} datasets")
    print(f"   - Combined file: non_crisis_combined.csv")
    print(f"\nNext step: Run phase4_combine/create_master_training_file.py")

else:
    print("\nNo datasets were successfully processed!")
    print("Check file paths and column names in the configuration.")
