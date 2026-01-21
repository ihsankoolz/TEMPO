"""
================================================================================
PHASE 4: COMBINE - Create Master Training File for Multi-Task BERT
================================================================================
Combines GoEmotions, Crisis, and Non-Crisis datasets into one master training
file with partial labels for multi-task BERT training.

Purpose:
    - Create unified dataset for training ONE multi-task BERT model
    - Handle partial labels (not all tweets have all 3 label types)
    - Shuffle data to prevent catastrophic forgetting

Input:
    - goemotion_data/goemotions.csv (emotion labels)
    - standardized_data/crisis_combined.csv (event type + informativeness)
    - standardized_data/non_crisis_combined.csv (event type)

Output:
    - master_training_data/master_training_data.csv
    - master_training_data/master_training_sample_1000.csv (for preview)

Master File Columns:
    text, emotion_fear, emotion_anger, emotion_joy, ...(13 emotions),
    event_type, informativeness, crisis_label, source_dataset, created_at

Partial Labels:
    - GoEmotions: Has emotions, NULL event_type/informativeness
    - Crisis: Has event_type/informativeness, NULL emotions
    - Non-Crisis: Has event_type, NULL emotions/informativeness

Note:
    Baseline dataset is NOT included - non-crisis provides sufficient contrast.

Usage:
    python scripts/phase4_combine/create_master_training_file.py

Prerequisites:
    - All Phase 3 scripts completed
    - standardized_data/ contains crisis_combined.csv and non_crisis_combined.csv

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("CREATING MASTER TRAINING FILE FOR MULTI-TASK BERT")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

GOEMOTIONS_PATH = "./goemotion_data/goemotions.csv"
CRISIS_COMBINED_PATH = "./standardized_data/crisis_combined.csv"
NON_CRISIS_COMBINED_PATH = "./standardized_data/non_crisis_combined.csv"

OUTPUT_DIR = "./master_training_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MASTER_FILE = os.path.join(OUTPUT_DIR, "master_training_data.csv")

# ============================================================================
# EMOTION LABEL MAPPING (from GoEmotions)
# ============================================================================

EMOTION_MAPPING = {
    2: 'anger',
    14: 'fear',
    25: 'sadness',
    19: 'nervousness',
    11: 'disgust',
    26: 'surprise',
    6: 'confusion',
    5: 'caring',
    16: 'grief',
    9: 'disappointment',
    17: 'joy',
    23: 'relief',
    27: 'neutral'
}

EMOTION_COLUMNS = [
    'emotion_fear', 'emotion_sadness', 'emotion_anger', 'emotion_nervousness',
    'emotion_disgust', 'emotion_surprise', 'emotion_confusion', 'emotion_caring',
    'emotion_grief', 'emotion_disappointment', 'emotion_joy', 'emotion_relief',
    'emotion_neutral'
]

# ============================================================================
# PROCESS GOEMOTIONS
# ============================================================================

def process_goemotions():
    print(f"\n{'='*80}")
    print("PROCESSING GOEMOTIONS")
    print(f"{'='*80}")

    if not os.path.exists(GOEMOTIONS_PATH):
        print(f"File not found: {GOEMOTIONS_PATH}")
        return None

    print(f"Loading: {GOEMOTIONS_PATH}")
    df = pd.read_csv(GOEMOTIONS_PATH)
    print(f"Loaded: {len(df):,} rows")

    print(f"Processing emotion labels...")

    def extract_emotions(labels_str):
        try:
            if pd.isna(labels_str):
                return {f'emotion_{name}': 0 for name in [
                    'fear', 'sadness', 'anger', 'nervousness', 'disgust',
                    'surprise', 'confusion', 'caring', 'grief', 'disappointment',
                    'joy', 'relief', 'neutral'
                ]}

            if isinstance(labels_str, str):
                labels = eval(labels_str)
            else:
                labels = labels_str

            return {
                'emotion_fear': 1 if 14 in labels else 0,
                'emotion_sadness': 1 if 25 in labels else 0,
                'emotion_anger': 1 if 2 in labels else 0,
                'emotion_nervousness': 1 if 19 in labels else 0,
                'emotion_disgust': 1 if 11 in labels else 0,
                'emotion_surprise': 1 if 26 in labels else 0,
                'emotion_confusion': 1 if 6 in labels else 0,
                'emotion_caring': 1 if 5 in labels else 0,
                'emotion_grief': 1 if 16 in labels else 0,
                'emotion_disappointment': 1 if 9 in labels else 0,
                'emotion_joy': 1 if 17 in labels else 0,
                'emotion_relief': 1 if 23 in labels else 0,
                'emotion_neutral': 1 if 27 in labels else 0,
            }
        except:
            return {f'emotion_{name}': 0 for name in [
                'fear', 'sadness', 'anger', 'nervousness', 'disgust',
                'surprise', 'confusion', 'caring', 'grief', 'disappointment',
                'joy', 'relief', 'neutral'
            ]}

    emotions = df['labels'].apply(extract_emotions)
    emotion_df = pd.DataFrame(emotions.tolist())

    print(f"\nEmotion Distribution (13 emotions):")
    for col in emotion_df.columns:
        count = emotion_df[col].sum()
        print(f"   {col:25s}: {count:,} tweets")

    standardized = pd.DataFrame({
        'text': df['text'],
        **{col: emotion_df[col] for col in EMOTION_COLUMNS},
        'event_type': None,
        'informativeness': None,
        'crisis_label': None,
        'source_dataset': 'goemotions',
        'created_at': None
    })

    before = len(standardized)
    standardized = standardized[standardized['text'].notna()]
    standardized = standardized[standardized['text'].str.strip() != '']
    after = len(standardized)

    if before > after:
        print(f"Removed {before - after:,} rows with missing/empty text")

    print(f"\nGoEmotions Processed: {len(standardized):,} rows")
    return standardized

# ============================================================================
# PROCESS CRISIS DATASETS
# ============================================================================

def process_crisis():
    print(f"\n{'='*80}")
    print("PROCESSING CRISIS DATASETS")
    print(f"{'='*80}")

    if not os.path.exists(CRISIS_COMBINED_PATH):
        print(f"File not found: {CRISIS_COMBINED_PATH}")
        return None

    print(f"Loading: {CRISIS_COMBINED_PATH}")
    df = pd.read_csv(CRISIS_COMBINED_PATH)
    print(f"Loaded: {len(df):,} rows")

    for emotion_col in EMOTION_COLUMNS:
        df[emotion_col] = None

    column_order = ['text'] + EMOTION_COLUMNS + ['event_type', 'informativeness',
                    'crisis_label', 'source_dataset', 'created_at']
    standardized = df[column_order]

    print(f"\nCrisis Data Processed: {len(standardized):,} rows")
    return standardized

# ============================================================================
# PROCESS NON-CRISIS DATASETS
# ============================================================================

def process_non_crisis():
    print(f"\n{'='*80}")
    print("PROCESSING NON-CRISIS DATASETS")
    print(f"{'='*80}")

    if not os.path.exists(NON_CRISIS_COMBINED_PATH):
        print(f"File not found: {NON_CRISIS_COMBINED_PATH}")
        return None

    print(f"Loading: {NON_CRISIS_COMBINED_PATH}")
    df = pd.read_csv(NON_CRISIS_COMBINED_PATH)
    print(f"Loaded: {len(df):,} rows")

    for emotion_col in EMOTION_COLUMNS:
        df[emotion_col] = None
    df['informativeness'] = None

    column_order = ['text'] + EMOTION_COLUMNS + ['event_type', 'informativeness',
                    'crisis_label', 'source_dataset', 'created_at']
    standardized = df[column_order]

    print(f"\nNon-Crisis Data Processed: {len(standardized):,} rows")
    return standardized

# ============================================================================
# COMBINE ALL DATASETS
# ============================================================================

def combine_all_datasets(goemotions_df, crisis_df, non_crisis_df):
    print(f"\n{'='*80}")
    print("COMBINING ALL DATASETS")
    print(f"{'='*80}")

    datasets = []
    dataset_names = []

    if goemotions_df is not None:
        datasets.append(goemotions_df)
        dataset_names.append('GoEmotions')
    if crisis_df is not None:
        datasets.append(crisis_df)
        dataset_names.append('Crisis')
    if non_crisis_df is not None:
        datasets.append(non_crisis_df)
        dataset_names.append('Non-Crisis')

    if len(datasets) == 0:
        print("No datasets to combine!")
        return None

    print(f"\nCombining {len(datasets)} datasets: {', '.join(dataset_names)}")

    master = pd.concat(datasets, ignore_index=True)
    print(f"Initial combine: {len(master):,} total rows")

    # CRITICAL: SHUFFLE THE DATA
    print(f"\nSHUFFLING DATA (critical for multi-task training)...")
    master = master.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Data shuffled!")

    # Display statistics
    print(f"\n{'='*80}")
    print("MASTER TRAINING FILE STATISTICS")
    print(f"{'='*80}")

    print(f"\nTOTAL ROWS: {len(master):,}")
    print(f"\nBY SOURCE DATASET:")
    print(master['source_dataset'].value_counts().to_string())
    print(f"\nBY EVENT TYPE:")
    print(master['event_type'].value_counts(dropna=False).to_string())

    print(f"\nLABEL AVAILABILITY:")
    print(f"   Emotion labels: {master['emotion_fear'].notna().sum():,} rows")
    print(f"   Event type labels: {master['event_type'].notna().sum():,} rows")
    print(f"   Informativeness labels: {master['informativeness'].notna().sum():,} rows")

    # Save master file
    print(f"\nSaving master training file...")
    master.to_csv(MASTER_FILE, index=False)
    print(f"SAVED: {MASTER_FILE}")

    # Save sample for preview
    sample_file = os.path.join(OUTPUT_DIR, "master_training_sample_1000.csv")
    master.head(1000).to_csv(sample_file, index=False)
    print(f"SAMPLE SAVED: {sample_file}")

    return master

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 1: PROCESSING INDIVIDUAL DATASETS")
    print("="*80)

    goemotions = process_goemotions()
    crisis = process_crisis()
    non_crisis = process_non_crisis()

    print("\n" + "="*80)
    print("STEP 2: COMBINING INTO MASTER FILE")
    print("="*80)

    master = combine_all_datasets(goemotions, crisis, non_crisis)

    if master is not None:
        print(f"\n{'='*80}")
        print("MASTER TRAINING FILE CREATED!")
        print(f"{'='*80}")
        print(f"""
Files created:
   - master_training_data.csv ({len(master):,} rows)
   - master_training_sample_1000.csv (for preview)

Datasets included:
   - GoEmotions (emotions)
   - Crisis datasets (event types + informativeness)
   - Non-crisis datasets (event types)

READY FOR MULTI-TASK BERT TRAINING!
        """)
