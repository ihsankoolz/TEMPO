"""
================================================================================
[OBSOLETE] Download CrisisLex - Old Version Without Timestamps
================================================================================
THIS SCRIPT IS OBSOLETE - Use scripts/phase1_download/download_crisislex.py instead

Reason for deprecation:
    - This version downloads ONLY the labeled tweets
    - It does NOT include timestamps (created_at)
    - Timestamps are CRITICAL for RL episode building

The new version (download_crisislex.py) merges:
    - *_labeled.csv (has tweet text + labels)
    - *_ids_and_timestamp.csv (has tweet_id + timestamp)

This achieves BOTH text content AND timestamps in one file.

Original Purpose:
    Downloads CrisisLex T26 dataset from GitHub (26 crisis events)

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import requests
import os

base_url = "https://raw.githubusercontent.com/sajao/CrisisLex/master/data/CrisisLexT26/"

# All 26 events
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

os.makedirs('crisislex_data', exist_ok=True)

all_tweets = []
successful_downloads = 0

for event in events:
    url = f"{base_url}{event}/{event}-tweets_labeled.csv"
    print(f"Downloading {event}...")

    try:
        df = pd.read_csv(url, encoding='utf-8', low_memory=False)
        print(f"  [OK] {len(df)} tweets")
        print(f"  Columns: {df.columns.tolist()[:5]}...")

        df['event_name'] = event
        df['source_dataset'] = 'CrisisLexT26'
        df['crisis_label'] = 1

        df.to_csv(f'crisislex_data/{event}.csv', index=False)
        all_tweets.append(df)
        successful_downloads += 1

    except Exception as e:
        print(f"  [FAILED]: {str(e)[:50]}...")

if all_tweets:
    combined = pd.concat(all_tweets, ignore_index=True)
    combined.to_csv('crisislex_all_combined.csv', index=False)
    print(f"\n[OK] SUCCESS!")
    print(f"Total events downloaded: {successful_downloads}/26")
    print(f"Total tweets: {len(combined)}")
    print(f"Saved to 'crisislex_all_combined.csv'")
else:
    print("\n[FAILED] No events downloaded successfully")
