"""
================================================================================
UTILITY: Check GoEmotions and Baseline Datasets
================================================================================
Pre-check script to run BEFORE create_master_training_file.py to:
1. Check GoEmotions format and see ALL 28 emotions
2. Check Baseline format and fix column headers
3. Decide which emotions to include for crisis detection

Purpose:
    - Validates GoEmotions dataset format and shows emotion distribution
    - Checks Baseline dataset format (handles missing headers)
    - Recommends which 13 emotions to use for crisis detection

Input:
    - goemotion_data/goemotions.csv
    - baseline_data/baseline_noise.csv (optional)

Output:
    - Console output showing emotion distribution and recommendations

Usage:
    python utils/check_goemotions_baseline.py

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os

print("="*80)
print("GOEMOTIONS & BASELINE PRE-CHECK")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

GOEMOTIONS_PATH = "./goemotion_data/goemotions.csv"
BASELINE_PATH = "./baseline_data/baseline_noise.csv"

# ============================================================================
# GOEMOTIONS: ALL 28 EMOTIONS
# ============================================================================

ALL_EMOTIONS = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral'
}

# ============================================================================
# CHECK GOEMOTIONS
# ============================================================================

def check_goemotions():
    """Explore GoEmotions dataset and show emotion distribution"""
    print(f"\n{'='*80}")
    print("CHECKING GOEMOTIONS")
    print(f"{'='*80}")

    if not os.path.exists(GOEMOTIONS_PATH):
        print(f"File not found: {GOEMOTIONS_PATH}")
        return None

    print(f"Loading: {GOEMOTIONS_PATH}")
    df = pd.read_csv(GOEMOTIONS_PATH)
    print(f"Loaded: {len(df):,} rows")

    print(f"\nColumns: {list(df.columns)}")

    print(f"\nSample Rows:")
    print(df.head(3).to_string())

    print(f"\nAnalyzing emotion distribution...")

    emotion_counts = {emotion: 0 for emotion in ALL_EMOTIONS.values()}

    for _, row in df.iterrows():
        try:
            labels = eval(row['labels']) if isinstance(row['labels'], str) else row['labels']
            for label_id in labels:
                if label_id in ALL_EMOTIONS:
                    emotion_counts[ALL_EMOTIONS[label_id]] += 1
        except:
            continue

    print(f"\nEMOTION DISTRIBUTION (All 28 emotions):")
    print(f"{'='*80}")

    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

    for emotion, count in sorted_emotions:
        percentage = (count / len(df)) * 100
        print(f"   {emotion:20s} : {count:6,} tweets ({percentage:5.2f}%)")

    print(f"\n{'='*80}")
    print("RECOMMENDED EMOTIONS FOR CRISIS DETECTION")
    print(f"{'='*80}")

    crisis_relevant = {
        'PRIMARY (Must Include)': [
            ('fear', 14, 'Direct indicator of danger/crisis'),
            ('sadness', 25, 'Grief, loss, despair'),
            ('anger', 2, 'Frustration, outrage at situation'),
            ('nervousness', 19, 'Anxiety, worry about crisis'),
            ('disgust', 11, 'Revulsion at disaster/violence'),
        ],
        'SECONDARY (Recommended)': [
            ('surprise', 26, 'Shock at unexpected event'),
            ('confusion', 6, 'Uncertainty, not knowing what to do'),
            ('caring', 5, 'Empathy, wanting to help victims'),
            ('grief', 16, 'Deep sadness, mourning'),
            ('disappointment', 9, 'Letdown, things getting worse'),
        ],
        'POSITIVE (Contrast - Optional)': [
            ('joy', 17, 'Happiness, celebration'),
            ('relief', 23, 'Crisis averted/ended'),
        ],
        'NEUTRAL': [
            ('neutral', 27, 'No strong emotion'),
        ]
    }

    for category, emotions in crisis_relevant.items():
        print(f"\n{category}:")
        for emotion, idx, description in emotions:
            count = emotion_counts.get(emotion, 0)
            print(f"   [{idx:2d}] {emotion:15s} - {description:40s} ({count:,} tweets)")

    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print("""
For crisis detection, the master training file uses 13 emotions:

PRIMARY (5): fear, sadness, anger, nervousness, disgust
SECONDARY (5): surprise, confusion, caring, grief, disappointment
POSITIVE (2): joy, relief
NEUTRAL (1): neutral

This gives BERT richer emotional features to learn from!
    """)

    return df

# ============================================================================
# CHECK BASELINE
# ============================================================================

def check_baseline():
    """Explore Baseline dataset and verify format"""
    print(f"\n{'='*80}")
    print("CHECKING BASELINE NOISE DATASET")
    print(f"{'='*80}")

    if not os.path.exists(BASELINE_PATH):
        print(f"File not found: {BASELINE_PATH}")
        print(f"   This dataset is optional - you can skip it")
        return None

    print(f"Loading: {BASELINE_PATH}")

    print(f"Checking if file has headers...")

    try:
        sample = pd.read_csv(BASELINE_PATH, nrows=1, header=None)
        first_row = sample.iloc[0].tolist()

        print(f"\nFirst row values:")
        print(f"   {first_row}")

        if isinstance(first_row[0], (int, float)) or str(first_row[0]).isdigit():
            print(f"\n[OK] CONFIRMED: File has NO headers (first value is a number: {first_row[0]})")
            has_headers = False
        else:
            print(f"\n[OK] File appears to have headers")
            has_headers = True

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    print(f"\nLoading full dataset...")

    if not has_headers:
        df = pd.read_csv(
            BASELINE_PATH,
            encoding='latin-1',
            header=None,
            names=['sentiment', 'id', 'date', 'query', 'user', 'text'],
            nrows=10000
        )
        print(f"Loaded: 10,000 rows (sample)")
        print(f"   Assigned column names: sentiment, id, date, query, user, text")
    else:
        df = pd.read_csv(BASELINE_PATH, encoding='latin-1', nrows=10000)
        print(f"Loaded: 10,000 rows (sample)")

    print(f"\nColumns: {list(df.columns)}")

    print(f"\nSample Rows:")
    print(df[['text', 'date', 'user']].head(3).to_string())

    print(f"\n[OK] Baseline format verified!")

    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    goemotions = check_goemotions()
    baseline = check_baseline()

    print(f"\n{'='*80}")
    print("PRE-CHECK COMPLETE")
    print(f"{'='*80}")
    print("""
Next steps:
1. Review emotion recommendations above
2. Run: python scripts/phase4_combine/create_master_training_file.py
3. Master file will have 13 emotion features!
    """)
