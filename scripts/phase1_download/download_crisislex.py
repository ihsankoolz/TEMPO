"""
================================================================================
PHASE 1: DOWNLOAD - CrisisLex T26 Dataset (WITH TIMESTAMPS)
================================================================================
Downloads the CrisisLex T26 dataset from GitHub. This script downloads BOTH
the labeled tweets AND the timestamp files, then merges them.

Purpose:
    - Provides crisis tweets with timestamps and informativeness labels
    - 28K tweets from 26 crisis events (2012-2013)
    - Includes informativeness labels (Related-Informative, Not Related, etc.)

Output:
    - crisis_datasets/crisislex_complete/ (individual event CSVs)
    - crisis_datasets/crisislex_all_complete.csv (combined file)

Events Included (26 total):
    - 2012 Colorado Wildfires, Hurricane Sandy
    - 2013 Boston Bombings, Oklahoma Tornadoes
    - Various earthquakes, floods, and other disasters

Key Columns:
    - Tweet Text, Informativeness, event_name, created_at (timestamp)

Usage:
    python scripts/phase1_download/download_crisislex.py

Note:
    This version merges labeled tweets WITH timestamps from the tweet-IDs file.
    The older version (download_crisislex_old.py) didn't include timestamps.

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import requests
import os

base_url = "https://raw.githubusercontent.com/sajao/CrisisLex/master/data/CrisisLexT26/"

events = [
    '2012_Colorado_wildfires',
    '2012_Costa_Rica_earthquake',
    '2012_Guatemala_earthquake',
    '2012_Italy_earthquakes',
    '2012_Philipinnes_floods',
    '2012_Typhoon_Pablo',
    '2012_Venezuela_refinery',
    '2013_Alberta_floods',
    '2013_Australia_bushfire',
    '2013_Bohol_earthquake',
    '2013_Boston_bombings',
    '2013_Brazil_nightclub_fire',
    '2013_Colorado_floods',
    '2013_LAX_shootings',
    '2013_Manila_floods',
    '2013_NY_train_crash',
    '2013_Oklahoma_tornadoes',
    '2013_Queensland_floods',
    '2013_Sardinia_floods',
    '2013_Savar_building_collapse',
    '2013_Singapore_haze',
    '2013_Spain_train_crash',
    '2013_Typhoon_Yolanda',
    '2013_West_Texas_explosion',
    '2012_Hurricane_Sandy',
    '2013_Russia_meteorite'
]

os.makedirs('crisis_datasets/crisislex_complete', exist_ok=True)

all_events = []
successful = 0
failed = 0

for event in events:
    print(f"\n{'='*60}")
    print(f"Processing: {event}")
    print('='*60)

    try:
        # Download File 1: Labeled tweets (text + labels)
        url_labeled = f"{base_url}{event}/{event}-tweets_labeled.csv"
        print(f"Downloading labeled tweets...")
        df_labeled = pd.read_csv(url_labeled, encoding='utf-8', low_memory=False)
        print(f"  {len(df_labeled)} labeled tweets")

        # Download File 2: Tweet IDs with timestamps
        url_ids = f"{base_url}{event}/{event}-tweetids_entire_period.csv"
        print(f"Downloading tweet IDs with timestamps...")
        df_ids = pd.read_csv(url_ids, encoding='utf-8', low_memory=False)
        print(f"  {len(df_ids)} tweet IDs with timestamps")

        # Show columns
        print(f"\nLabeled file columns: {df_labeled.columns.tolist()}")
        print(f"IDs file columns: {df_ids.columns.tolist()}")

        # Clean column names (remove extra spaces)
        df_labeled.columns = df_labeled.columns.str.strip()
        df_ids.columns = df_ids.columns.str.strip()

        # Find the tweet ID column name (might have different names)
        id_col_labeled = [col for col in df_labeled.columns if 'id' in col.lower()][0]
        id_col_ids = [col for col in df_ids.columns if 'id' in col.lower()][0]

        print(f"\nMerging on: {id_col_labeled} (labeled) <-> {id_col_ids} (ids)")

        # Merge on tweet ID
        df_merged = df_labeled.merge(
            df_ids,
            left_on=id_col_labeled,
            right_on=id_col_ids,
            how='inner'  # Only keep tweets that have both text AND timestamp
        )

        print(f"  Merged: {len(df_merged)} tweets with BOTH text and timestamps")

        # Add metadata
        df_merged['event_name'] = event
        df_merged['source_dataset'] = 'CrisisLexT26'
        df_merged['crisis_label'] = 1

        # Check for timestamp column
        timestamp_col = [col for col in df_merged.columns if 'time' in col.lower()][0]
        print(f"  Timestamp column: {timestamp_col}")

        # Convert timestamp
        df_merged['created_at'] = pd.to_datetime(df_merged[timestamp_col])

        # Show date range
        print(f"  Date range: {df_merged['created_at'].min()} to {df_merged['created_at'].max()}")

        # Save
        output_file = f'crisis_datasets/crisislex_complete/{event}_complete.csv'
        df_merged.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

        all_events.append(df_merged)
        successful += 1

    except Exception as e:
        print(f"  Failed: {str(e)[:100]}...")
        failed += 1

# Combine all events
if all_events:
    print(f"\n{'='*60}")
    print("COMBINING ALL EVENTS")
    print('='*60)

    combined = pd.concat(all_events, ignore_index=True)
    combined.to_csv('crisis_datasets/crisislex_all_complete.csv', index=False)

    print(f"\nSUCCESS!")
    print(f"Total events downloaded: {successful}/26")
    print(f"Failed: {failed}/26")
    print(f"Total tweets: {len(combined)}")
    print(f"Date range: {combined['created_at'].min()} to {combined['created_at'].max()}")
    print(f"\nSaved to: crisis_datasets/crisislex_all_complete.csv")

else:
    print("\nNo events downloaded successfully")
