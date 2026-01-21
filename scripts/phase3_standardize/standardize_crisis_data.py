"""
================================================================================
PHASE 3: STANDARDIZE - Crisis Datasets (HumAID & CrisisLex)
================================================================================
Standardizes HumAID and CrisisLex datasets to a common format and maps
event names to general event types for BERT training.

Purpose:
    - Unify column headers across crisis datasets
    - Map specific event names (e.g., 'hurricane_harvey_2017') to types ('hurricane')
    - Prepare data for multi-task BERT training

Input:
    - crisis_datasets/humaid_all_with_timestamps.csv
    - crisis_datasets/crisislex_all_complete.csv

Output:
    - standardized_data/humaid_standardized.csv
    - standardized_data/crisislex_standardized.csv
    - standardized_data/crisis_combined.csv

Standard Columns:
    text, created_at, event_name, event_type, crisis_label, source_dataset, informativeness

Event Type Mapping:
    - hurricane, earthquake, flood, wildfire, shooting, bombing, etc.

Usage:
    python scripts/phase3_standardize/standardize_crisis_data.py

Prerequisites:
    - Run phase2_process scripts first
    - Run utils/check_crisis_events.py to verify mappings

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os

print("="*80)
print("STANDARDIZING CRISIS DATASETS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

HUMAID_PATH = "./crisis_datasets/humaid_all_with_timestamps.csv"
CRISISLEX_PATH = "./crisis_datasets/crisislex_all_complete.csv"
OUTPUT_DIR = "./standardized_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# EVENT TYPE MAPPING
# ============================================================================

EVENT_TYPE_MAPPING = {
    # Weather disasters
    'hurricane': ['hurricane', 'cyclone', 'typhoon', 'storm'],
    'flood': ['flood', 'flooding'],
    'tornado': ['tornado'],
    'wildfire': ['wildfire', 'fire', 'bushfire', 'forestfire', 'wildland'],
    'haze': ['haze'],

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
    'accident': ['accident', 'crash', 'collision', 'derailment', 'refinery', 'collapse', 'building-collapse'],

    # Disease outbreaks
    'disease_outbreak': ['covid', 'corona', 'virus', 'pandemic', 'epidemic',
                         'outbreak', 'ebola', 'zika', 'flu', 'disease'],

    # Other
    'drought': ['drought'],
    'heatwave': ['heatwave', 'heat wave'],
    'sinkhole': ['sinkhole'],
}

def map_event_to_type(event_name):
    """Map a specific event name to a general event type"""
    event_name_lower = event_name.lower()

    for event_type, keywords in EVENT_TYPE_MAPPING.items():
        for keyword in keywords:
            if keyword in event_name_lower:
                return event_type

    return 'other_crisis'

# ============================================================================
# STANDARDIZE HUMAID
# ============================================================================

def standardize_humaid():
    print(f"\n{'='*80}")
    print("PROCESSING HUMAID")
    print(f"{'='*80}")

    if not os.path.exists(HUMAID_PATH):
        print(f"File not found: {HUMAID_PATH}")
        return None

    print(f"Loading: {HUMAID_PATH}")
    humaid = pd.read_csv(HUMAID_PATH, low_memory=False)
    print(f"Loaded: {len(humaid):,} rows")
    print(f"Columns: {list(humaid.columns)}")

    # Map event names to types
    print(f"Mapping event names to event types...")
    humaid['event_type'] = humaid['event_name'].apply(map_event_to_type)

    print(f"\nEvent Type Distribution:")
    print(humaid['event_type'].value_counts().to_string())

    # Standardize to common format
    standardized = pd.DataFrame({
        'text': humaid['tweet_text'],
        'created_at': pd.to_datetime(humaid['created_at'], format='mixed', errors='coerce'),
        'event_name': humaid['event_name'],
        'event_type': humaid['event_type'],
        'crisis_label': 1,
        'source_dataset': 'humaid',
        'informativeness': None
    })

    # Remove rows with missing text or timestamp
    before = len(standardized)
    standardized = standardized.dropna(subset=['text', 'created_at'])
    after = len(standardized)
    if before > after:
        print(f"Removed {before - after:,} rows with missing text/timestamp")

    # Remove duplicates
    before = len(standardized)
    standardized = standardized.drop_duplicates(subset=['text', 'created_at'])
    after = len(standardized)
    if before > after:
        print(f"Removed {before - after:,} duplicate tweets")

    print(f"\nHumAID Standardized: {len(standardized):,} rows")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "humaid_standardized.csv")
    standardized.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return standardized

# ============================================================================
# STANDARDIZE CRISISLEX
# ============================================================================

def standardize_crisislex():
    print(f"\n{'='*80}")
    print("PROCESSING CRISISLEX")
    print(f"{'='*80}")

    if not os.path.exists(CRISISLEX_PATH):
        print(f"File not found: {CRISISLEX_PATH}")
        return None

    print(f"Loading: {CRISISLEX_PATH}")
    crisislex = pd.read_csv(CRISISLEX_PATH, low_memory=False)
    print(f"Loaded: {len(crisislex):,} rows")
    print(f"Columns: {list(crisislex.columns)}")

    # Map event names to types
    print(f"Mapping event names to event types...")
    crisislex['event_type'] = crisislex['event_name'].apply(map_event_to_type)

    print(f"\nEvent Type Distribution:")
    print(crisislex['event_type'].value_counts().to_string())

    # Clean informativeness labels
    print(f"\nInformativeness Distribution (before cleaning):")
    print(crisislex['Informativeness'].value_counts().to_string())

    def clean_informativeness(label):
        if pd.isna(label):
            return None
        label = str(label).strip().lower()

        if 'not related' in label or 'notrelated' in label:
            return 'not_related'
        elif 'not informative' in label or 'not-informative' in label:
            return 'related_not_informative'
        elif 'informative' in label:
            return 'related_informative'
        elif 'not labeled' in label:
            return None
        else:
            return None

    crisislex['informativeness_clean'] = crisislex['Informativeness'].apply(clean_informativeness)

    print(f"\nInformativeness Distribution (after cleaning):")
    print(crisislex['informativeness_clean'].value_counts(dropna=False).to_string())

    # Standardize to common format
    standardized = pd.DataFrame({
        'text': crisislex['Tweet Text'],
        'created_at': pd.to_datetime(crisislex['created_at'], format='mixed', errors='coerce'),
        'event_name': crisislex['event_name'],
        'event_type': crisislex['event_type'],
        'crisis_label': 1,
        'source_dataset': 'crisislex',
        'informativeness': crisislex['informativeness_clean']
    })

    # Remove rows with missing text or timestamp
    before = len(standardized)
    standardized = standardized.dropna(subset=['text', 'created_at'])
    after = len(standardized)
    if before > after:
        print(f"Removed {before - after:,} rows with missing text/timestamp")

    # Remove duplicates
    before = len(standardized)
    standardized = standardized.drop_duplicates(subset=['text', 'created_at'])
    after = len(standardized)
    if before > after:
        print(f"Removed {before - after:,} duplicate tweets")

    print(f"\nCrisisLex Standardized: {len(standardized):,} rows")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "crisislex_standardized.csv")
    standardized.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return standardized

# ============================================================================
# COMBINE CRISIS DATASETS
# ============================================================================

def combine_crisis_datasets(humaid_df, crisislex_df):
    print(f"\n{'='*80}")
    print("COMBINING CRISIS DATASETS")
    print(f"{'='*80}")

    if humaid_df is None or crisislex_df is None:
        print("Cannot combine - one or both datasets failed to load")
        return None

    combined = pd.concat([humaid_df, crisislex_df], ignore_index=True)

    print(f"\nCOMBINED CRISIS DATASET SUMMARY:")
    print(f"   Total tweets: {len(combined):,}")
    print(f"\n   By Event Type:")
    print(combined['event_type'].value_counts().to_string())
    print(f"\n   By Source:")
    print(combined['source_dataset'].value_counts().to_string())
    print(f"\n   Date Range:")
    print(f"   Earliest: {combined['created_at'].min()}")
    print(f"   Latest: {combined['created_at'].max()}")

    # Save combined
    output_path = os.path.join(OUTPUT_DIR, "crisis_combined.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nCOMBINED FILE SAVED: {output_path}")

    return combined

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    humaid_std = standardize_humaid()
    crisislex_std = standardize_crisislex()
    crisis_combined = combine_crisis_datasets(humaid_std, crisislex_std)

    if crisis_combined is not None:
        print(f"\n{'='*80}")
        print("CRISIS DATASETS STANDARDIZED!")
        print(f"{'='*80}")
        print(f"\nFiles created:")
        print(f"   - humaid_standardized.csv")
        print(f"   - crisislex_standardized.csv")
        print(f"   - crisis_combined.csv")
        print(f"\nNext step: Run phase4_combine/create_master_training_file.py")
