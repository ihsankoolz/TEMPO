"""
================================================================================
PHASE 1: DOWNLOAD - GoEmotions Dataset
================================================================================
Downloads the GoEmotions dataset from Hugging Face for emotion classification
training. GoEmotions contains 58K Reddit comments labeled with 27 emotions.

Purpose:
    - Pre-train BERT to recognize emotions (fear, anger, joy, sadness, etc.)
    - Provides emotion labels for the multi-task BERT model

Output:
    - goemotion_data/goemotions.csv (or goemotions_full.csv)

Usage:
    python scripts/phase1_download/download_goemotions.py

Dependencies:
    pip install datasets pandas

Author: JIKI DAP Round 2 Team
Date: January 2026
================================================================================
"""

from datasets import load_dataset
import pandas as pd
import os

print("="*70)
print("DOWNLOADING GOEMOTIONS - EMOTION TRAINING")
print("="*70)

# Create output directory
os.makedirs('goemotion_data', exist_ok=True)

# Download from Hugging Face
print("\nDownloading GoEmotions (58K labeled emotions)...")
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

print("Download complete!")

# Convert to pandas
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])

# Combine all splits
combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\nTotal rows: {len(combined):,}")
print(f"Columns: {combined.columns.tolist()}")
print(f"\nFirst example:")
print(combined.iloc[0])

# Save
output_path = 'goemotion_data/goemotions.csv'
combined.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Show emotion distribution
print(f"\n{'='*70}")
print("EMOTION CATEGORIES")
print('='*70)

# GoEmotions has 27 emotions + neutral
emotion_cols = [col for col in combined.columns if col not in ['text', 'id']]
print(f"Available emotions: {emotion_cols[:10]}...")  # Show first 10
print(f"Total emotion categories: {len(emotion_cols)}")

print(f"\nGoEmotions dataset ready for BERT emotion training!")
