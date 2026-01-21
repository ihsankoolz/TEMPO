"""
================================================================================
UTILITY: Check Crisis Events and Verify Mappings
================================================================================
Pre-check script to run BEFORE standardize_crisis_data.py to:
1. See all unique events in your crisis datasets
2. Check which events don't have type mappings
3. Verify all important columns are present

Purpose:
    - Validates HumAID and CrisisLex datasets are properly formatted
    - Shows which events will map to which event types
    - Identifies unmapped events that need to be added to EVENT_TYPE_MAPPING

Input:
    - crisis_datasets/humaid_all_with_timestamps.csv
    - crisis_datasets/crisislex_all_complete.csv

Output:
    - Console output showing event mappings and recommendations

Usage:
    python utils/check_crisis_events.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os
from collections import defaultdict

print("="*80)
print("CRISIS DATASET PRE-CHECK")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

HUMAID_PATH = "./crisis_datasets/humaid_all_with_timestamps.csv"
CRISISLEX_PATH = "./crisis_datasets/crisislex_all_complete.csv"

# ============================================================================
# CURRENT EVENT TYPE MAPPING (from standardize script)
# ============================================================================

EVENT_TYPE_MAPPING = {
    # Weather disasters
    'hurricane': ['hurricane', 'cyclone', 'typhoon', 'storm'],
    'flood': ['flood', 'flooding'],
    'tornado': ['tornado'],
    'wildfire': ['wildfire', 'fire', 'bushfire', 'forestfire', 'wildland'],

    # Geological disasters
    'earthquake': ['earthquake', 'quake', 'tremor'],
    'tsunami': ['tsunami'],
    'landslide': ['landslide', 'mudslide'],
    'avalanche': ['avalanche'],

    # Violence/Attacks
    'shooting': ['shooting', 'shoot', 'gunfire'],
    'bombing': ['bombing', 'bomb', 'explosion', 'blast'],
    'attack': ['attack', 'assault', 'terror'],

    # Civil unrest
    'protest': ['protest', 'riot', 'demonstration', 'unrest'],

    # Accidents
    'accident': ['accident', 'crash', 'collision', 'derailment'],

    # Disease outbreaks
    'disease_outbreak': ['covid', 'corona', 'virus', 'pandemic', 'epidemic',
                        'outbreak', 'ebola', 'zika', 'flu', 'disease'],

    # Other specific disasters
    'drought': ['drought'],
    'heatwave': ['heatwave', 'heat wave'],
    'sinkhole': ['sinkhole'],
}

def map_event_to_type(event_name):
    """Map event name to type (same logic as standardize script)"""
    event_name_lower = event_name.lower()

    for event_type, keywords in EVENT_TYPE_MAPPING.items():
        for keyword in keywords:
            if keyword in event_name_lower:
                return event_type

    return 'other_crisis'

# ============================================================================
# CHECK HUMAID
# ============================================================================

def check_humaid():
    """Explore HumAID dataset"""
    print(f"\n{'='*80}")
    print("CHECKING HUMAID")
    print(f"{'='*80}")

    if not os.path.exists(HUMAID_PATH):
        print(f"File not found: {HUMAID_PATH}")
        return None

    print(f"Loading: {HUMAID_PATH}")
    df = pd.read_csv(HUMAID_PATH, low_memory=False)
    print(f"Loaded: {len(df):,} rows")

    print(f"\nCOLUMN CHECK:")
    required_cols = ['tweet_text', 'created_at', 'event_name', 'crisis_label']
    print(f"   Required columns: {required_cols}")

    for col in required_cols:
        if col in df.columns:
            print(f"   [OK] {col} - PRESENT")
        else:
            print(f"   [MISSING] {col} - MISSING!")
            print(f"      Available columns: {list(df.columns)}")
            return None

    events = df['event_name'].unique()
    print(f"\nUNIQUE EVENTS IN HUMAID:")
    print(f"   Total unique events: {len(events)}")

    event_mapping_results = {}
    unmapped = []

    for event in sorted(events):
        mapped_type = map_event_to_type(event)
        event_mapping_results[event] = mapped_type

        if mapped_type == 'other_crisis':
            unmapped.append(event)

    print(f"\nEVENT -> TYPE MAPPING:")

    type_groups = defaultdict(list)
    for event, event_type in event_mapping_results.items():
        type_groups[event_type].append(event)

    for event_type in sorted(type_groups.keys()):
        print(f"\n   {event_type.upper()}:")
        for event in sorted(type_groups[event_type]):
            count = len(df[df['event_name'] == event])
            print(f"      - {event} ({count:,} tweets)")

    if unmapped:
        print(f"\n[WARNING] UNMAPPED EVENTS (will be labeled 'other_crisis'):")
        for event in sorted(unmapped):
            count = len(df[df['event_name'] == event])
            print(f"      - {event} ({count:,} tweets)")
        print(f"\n   Add these to EVENT_TYPE_MAPPING if they need specific types!")
    else:
        print(f"\n[OK] All events have mappings!")

    return df

# ============================================================================
# CHECK CRISISLEX
# ============================================================================

def check_crisislex():
    """Explore CrisisLex dataset"""
    print(f"\n{'='*80}")
    print("CHECKING CRISISLEX")
    print(f"{'='*80}")

    if not os.path.exists(CRISISLEX_PATH):
        print(f"File not found: {CRISISLEX_PATH}")
        return None

    print(f"Loading: {CRISISLEX_PATH}")
    df = pd.read_csv(CRISISLEX_PATH, low_memory=False)
    print(f"Loaded: {len(df):,} rows")

    print(f"\nCOLUMN CHECK:")
    required_cols = ['Tweet Text', 'event_name', 'created_at', 'crisis_label', 'Informativeness']
    print(f"   Required columns: {required_cols}")

    for col in required_cols:
        if col in df.columns:
            print(f"   [OK] {col} - PRESENT")

            if col == 'Informativeness':
                print(f"\n   Informativeness Label Distribution:")
                print(df['Informativeness'].value_counts().to_string())
        else:
            print(f"   [MISSING] {col} - MISSING!")
            print(f"      Available columns: {list(df.columns)}")
            return None

    events = df['event_name'].unique()
    print(f"\nUNIQUE EVENTS IN CRISISLEX:")
    print(f"   Total unique events: {len(events)}")

    event_mapping_results = {}
    unmapped = []

    for event in sorted(events):
        mapped_type = map_event_to_type(event)
        event_mapping_results[event] = mapped_type

        if mapped_type == 'other_crisis':
            unmapped.append(event)

    print(f"\nEVENT -> TYPE MAPPING:")

    type_groups = defaultdict(list)
    for event, event_type in event_mapping_results.items():
        type_groups[event_type].append(event)

    for event_type in sorted(type_groups.keys()):
        print(f"\n   {event_type.upper()}:")
        for event in sorted(type_groups[event_type]):
            count = len(df[df['event_name'] == event])
            print(f"      - {event} ({count:,} tweets)")

    if unmapped:
        print(f"\n[WARNING] UNMAPPED EVENTS (will be labeled 'other_crisis'):")
        for event in sorted(unmapped):
            count = len(df[df['event_name'] == event])
            print(f"      - {event} ({count:,} tweets)")
        print(f"\n   Add these to EVENT_TYPE_MAPPING if they need specific types!")
    else:
        print(f"\n[OK] All events have mappings!")

    return df

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

def generate_recommendations(humaid_df, crisislex_df):
    """Generate recommendations for event mappings"""
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")

    if humaid_df is None and crisislex_df is None:
        print("No datasets loaded - check file paths!")
        return

    all_unmapped = set()

    if humaid_df is not None:
        humaid_events = humaid_df['event_name'].unique()
        for event in humaid_events:
            if map_event_to_type(event) == 'other_crisis':
                all_unmapped.add(event)

    if crisislex_df is not None:
        crisislex_events = crisislex_df['event_name'].unique()
        for event in crisislex_events:
            if map_event_to_type(event) == 'other_crisis':
                all_unmapped.add(event)

    if all_unmapped:
        print(f"\n[WARNING] FOUND {len(all_unmapped)} UNMAPPED EVENTS")
        print(f"\nRECOMMENDED ACTION:")
        print(f"   Update EVENT_TYPE_MAPPING in standardize_crisis_data.py")
    else:
        print(f"\n[OK] ALL EVENTS HAVE MAPPINGS!")
        print(f"   You can proceed with standardize_crisis_data.py")

    total_tweets = 0
    if humaid_df is not None:
        total_tweets += len(humaid_df)
    if crisislex_df is not None:
        total_tweets += len(crisislex_df)

    print(f"\nTOTAL CRISIS TWEETS: {total_tweets:,}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\nThis script checks your crisis datasets BEFORE standardization.\n")

    humaid = check_humaid()
    crisislex = check_crisislex()

    generate_recommendations(humaid, crisislex)

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("""
1. Review the unmapped events above (if any)
2. Update EVENT_TYPE_MAPPING in standardize_crisis_data.py if needed
3. Run: python scripts/phase3_standardize/standardize_crisis_data.py
4. Proceed with master file creation
    """)
