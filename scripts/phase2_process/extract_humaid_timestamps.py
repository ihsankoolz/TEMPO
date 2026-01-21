"""
================================================================================
PHASE 2: PROCESS - Extract Timestamps from HumAID Tweet IDs
================================================================================
Extracts timestamps from HumAID tweet IDs using Twitter's Snowflake ID algorithm.
Twitter IDs encode the creation timestamp in the ID itself.

Purpose:
    - Convert tweet IDs to actual timestamps
    - Essential for building temporal episodes for RL training
    - Enables multi-timescale features (1hr ago, 3hr ago, 6hr ago)

Input:
    - crisis_datasets/humaid_crisis_data/*.tsv (raw HumAID files)

Output:
    - crisis_datasets/humaid_crisis_data/*_with_timestamps.csv

Algorithm:
    Twitter Snowflake ID format:
    - Bits 63-22: Timestamp (milliseconds since Twitter epoch)
    - Twitter epoch: 2010-11-04 01:42:54 UTC (1288834974657 ms)
    - timestamp_ms = (tweet_id >> 22) + 1288834974657

Usage:
    python scripts/phase2_process/extract_humaid_timestamps.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os
from datetime import datetime, timezone

def snowflake_to_timestamp(tweet_id):
    """
    Convert Twitter Snowflake ID to timestamp.
    Twitter epoch starts at 2010-11-04 01:42:54 UTC
    """
    twitter_epoch = 1288834974657  # milliseconds
    timestamp_ms = (tweet_id >> 22) + twitter_epoch
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

# Get all TSV files
files = []
for root, dirs, filenames in os.walk('crisis_datasets/humaid_crisis_data/'):
    for filename in filenames:
        if filename.endswith('.tsv'):
            files.append(os.path.join(root, filename))

print(f"Found {len(files)} event files")

for file_path in files:
    print(f"\nProcessing {os.path.basename(file_path)}...")

    # Read file
    df = pd.read_csv(file_path, sep='\t')

    # Extract timestamps from tweet IDs
    df['created_at'] = df['tweet_id'].apply(snowflake_to_timestamp)

    # Show results
    print(f"  Tweets: {len(df)}")
    print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"  Sample timestamps:")
    print(df[['tweet_id', 'created_at']].head(3))

    # Save with timestamps
    output_path = file_path.replace('.tsv', '_with_timestamps.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to {os.path.basename(output_path)}")

print("\nAll files processed with timestamps!")
print("\nNext step: Run phase2_process/combine_humaid_files.py")
