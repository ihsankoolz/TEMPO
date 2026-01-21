"""
================================================================================
PHASE 2: PROCESS - Combine All HumAID Files
================================================================================
Combines all individual HumAID event files (with timestamps) into a single
master HumAID file for easier processing.

Purpose:
    - Merge train/dev/test splits for each event
    - Create one unified HumAID dataset
    - Add event_name and source_dataset metadata

Input:
    - crisis_datasets/humaid_crisis_data/*_with_timestamps.csv

Output:
    - crisis_datasets/humaid_all_with_timestamps.csv

Usage:
    python scripts/phase2_process/combine_humaid_files.py

Prerequisites:
    - Run extract_humaid_timestamps.py first

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os
import glob

print("Combining all HumAID files with timestamps...")

# Find all timestamp files
pattern = 'crisis_datasets/humaid_crisis_data/**/*_with_timestamps.csv'
timestamp_files = glob.glob(pattern, recursive=True)

print(f"Found {len(timestamp_files)} files with timestamps")

# Combine all files
all_data = []

for file_path in timestamp_files:
    event_name = os.path.basename(file_path).replace('_with_timestamps.csv', '')

    df = pd.read_csv(file_path)
    df['event_name'] = event_name
    df['source_dataset'] = 'HumAID'
    df['crisis_label'] = 1

    all_data.append(df)
    print(f"  {event_name}: {len(df)} tweets")

# Combine
combined = pd.concat(all_data, ignore_index=True)

# Convert timestamp column to datetime - use format='mixed' for different formats
combined['created_at'] = pd.to_datetime(combined['created_at'], format='mixed')

print(f"\n{'='*60}")
print("COMBINED HUMAID DATASET")
print('='*60)
print(f"Total tweets: {len(combined)}")
print(f"Unique events: {combined['event_name'].nunique()}")
print(f"Date range: {combined['created_at'].min()} to {combined['created_at'].max()}")
print(f"\nColumns: {combined.columns.tolist()}")

# Save combined file
output_path = 'crisis_datasets/humaid_all_with_timestamps.csv'
combined.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Show some stats
print(f"\n{'='*60}")
print("EVENT BREAKDOWN")
print('='*60)

# Group by event (removing _train/_test/_dev suffix)
combined['base_event'] = combined['event_name'].str.replace('_train|_test|_dev', '', regex=True)
event_counts = combined.groupby('base_event').size().sort_values(ascending=False)

print("\nTweets per event:")
for event, count in event_counts.items():
    print(f"  {event}: {count} tweets")

# Show hourly distribution for Hurricane Harvey
print(f"\n{'='*60}")
print("EXAMPLE: Hurricane Harvey hourly breakdown")
print('='*60)
harvey = combined[combined['event_name'].str.contains('harvey', case=False)].copy()
harvey['hour'] = harvey['created_at'].dt.floor('H')
hourly = harvey.groupby('hour').size().sort_index()

print(f"\nTotal Hurricane Harvey tweets: {len(harvey)}")
print(f"Date range: {harvey['created_at'].min()} to {harvey['created_at'].max()}")
print(f"\nFirst 24 hours of activity:")
print(hourly.head(24))

print(f"\nALL HUMAID DATA COMBINED SUCCESSFULLY!")
print("\nNext step: Run phase3_standardize/standardize_crisis_data.py")
