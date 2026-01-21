"""
================================================================================
[OBSOLETE] Test Script - Hurricane Harvey Data Inspection
================================================================================
THIS SCRIPT IS OBSOLETE - Used for one-time testing only

Purpose:
    Quick test script to verify Hurricane Harvey timestamp extraction worked.
    Used during development to check hourly tweet distribution.

This was a development/debugging script, not part of the main pipeline.

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os

# Pick one event to inspect
event_file = 'humaid_crisis_data/hurricane_harvey_2017_train_with_timestamps.csv'

print("Checking Hurricane Harvey data...")
df = pd.read_csv(event_file)

print(f"\nTotal tweets: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDate range: {df['created_at'].min()} to {df['created_at'].max()}")

# Show hourly distribution
df['created_at'] = pd.to_datetime(df['created_at'])
df['hour'] = df['created_at'].dt.floor('H')
hourly = df.groupby('hour').size()

print(f"\nTweets per hour (first 10 hours):")
print(hourly.head(10))

print(f"\n[OK] Success! You now have {len(df)} Hurricane Harvey tweets with timestamps")
