"""
================================================================================
UTILITY: Inspect Large Dataset Statistics
================================================================================
Use this to inspect CSV files that are too large for Excel.
Shows row counts, column info, sample data, and full distribution analysis.

Purpose:
    - View statistics of large CSV files without loading into Excel
    - Check if file exceeds Excel's 1M row limit
    - Preview data structure and distributions
    - Analyze event distribution using memory-efficient chunking

Input:
    - Any CSV file (default: standardized_data/non_crisis_combined.csv)

Output:
    - Console output showing file statistics, samples, and distributions

Usage:
    python utils/inspect_large_dataset.py

Note:
    Edit the file_path variable to inspect different files.

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

import pandas as pd
import os

print("="*80)
print("LARGE DATASET INSPECTOR")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File to inspect (UPDATE THIS PATH)
file_path = "./standardized_data/non_crisis_combined.csv"

# ============================================================================
# INSPECTION FUNCTIONS
# ============================================================================

def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def count_rows_fast(filepath):
    """Count rows without loading entire file into memory"""
    print("Counting rows (this may take a moment for large files)...")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        row_count = sum(1 for line in f) - 1  # Subtract 1 for header

    return row_count

def inspect_dataset(filepath):
    """Inspect a large CSV file without loading it all into memory"""

    print(f"\nFile: {filepath}")
    print(f"{'='*80}\n")

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print(f"   Please update the 'file_path' variable in the script")
        return

    size_mb = get_file_size(filepath)
    print(f"File Size: {size_mb:.2f} MB")

    total_rows = count_rows_fast(filepath)
    print(f"Total Rows: {total_rows:,}")

    excel_limit = 1_048_576
    if total_rows > excel_limit:
        print(f"[WARNING] EXCEEDS EXCEL LIMIT ({excel_limit:,} rows)")
        print(f"   Overflow: {total_rows - excel_limit:,} rows won't fit in Excel")
        print(f"   This is GOOD for ML! More data = better model")
    else:
        print(f"[OK] Within Excel limit ({excel_limit:,} rows)")

    print(f"\n{'='*80}")
    print("LOADING SAMPLE DATA (First 1000 rows for preview)")
    print(f"{'='*80}\n")

    try:
        sample_df = pd.read_csv(filepath, nrows=1000)

        print(f"Columns ({len(sample_df.columns)}):")
        print(f"   {list(sample_df.columns)}\n")

        print(f"Data Types:")
        print(sample_df.dtypes.to_string())
        print()

        missing = sample_df.isnull().sum()
        if missing.any():
            print(f"Missing Values (in first 1000 rows):")
            print(missing[missing > 0].to_string())
            print()

        if 'event_name' in sample_df.columns:
            print(f"Event Distribution (in sample):")
            print(sample_df['event_name'].value_counts().to_string())
            print()

        if 'event_type' in sample_df.columns:
            print(f"Event Type Distribution (in sample):")
            print(sample_df['event_type'].value_counts().to_string())
            print()

        if 'created_at' in sample_df.columns:
            sample_df['created_at'] = pd.to_datetime(sample_df['created_at'])
            print(f"Date Range (in sample):")
            print(f"   Earliest: {sample_df['created_at'].min()}")
            print(f"   Latest: {sample_df['created_at'].max()}")
            print()

        print(f"{'='*80}")
        print("PREVIEW: First 5 Rows")
        print(f"{'='*80}\n")
        preview_cols = ['text', 'event_name', 'event_type', 'created_at']
        available_cols = [c for c in preview_cols if c in sample_df.columns]
        print(sample_df[available_cols].head().to_string())

    except Exception as e:
        print(f"Error loading sample: {e}")
        return

    print(f"\n{'='*80}")
    print("ESTIMATING FULL DATASET DISTRIBUTION")
    print(f"{'='*80}\n")
    print("Loading with chunking to count by event...")

    try:
        event_counts = {}
        event_type_counts = {}

        chunk_size = 100000
        chunks_processed = 0

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            if 'event_name' in chunk.columns:
                for event in chunk['event_name'].value_counts().items():
                    event_counts[event[0]] = event_counts.get(event[0], 0) + event[1]

            if 'event_type' in chunk.columns:
                for etype in chunk['event_type'].value_counts().items():
                    event_type_counts[etype[0]] = event_type_counts.get(etype[0], 0) + etype[1]

            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f"   Processed {chunks_processed * chunk_size:,} rows...")

        print(f"\n[OK] Full dataset analysis complete!\n")

        if event_counts:
            print(f"FULL DATASET - Events:")
            event_df = pd.DataFrame(list(event_counts.items()), columns=['Event', 'Count'])
            event_df = event_df.sort_values('Count', ascending=False)
            print(event_df.to_string(index=False))
            print(f"\n   Total Events: {len(event_counts)}")

        print()

        if event_type_counts:
            print(f"FULL DATASET - Event Types:")
            etype_df = pd.DataFrame(list(event_type_counts.items()), columns=['Type', 'Count'])
            etype_df = etype_df.sort_values('Count', ascending=False)
            print(etype_df.to_string(index=False))

    except Exception as e:
        print(f"Error analyzing full dataset: {e}")

    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}\n")
    print(f"TIP: For ML training, you'll use pandas.read_csv() in Python")
    print(f"   You don't need to open this in Excel!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    inspect_dataset(file_path)

    print(f"\n{'='*80}")
    print("NOTES")
    print(f"{'='*80}")
    print("""
    Your dataset is ready for BERT training!

    Large file size = More training data = Better model

    You don't need Excel for ML work. Use Python scripts like this one.

    Next steps:
       1. Combine with crisis datasets (HumAID + CrisisLex)
       2. Add GoEmotions for emotion labels
       3. Create multi-task BERT training file
    """)
