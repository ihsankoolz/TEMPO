"""
Impute missing timestamps for datasets without date information.

This module provides utilities for handling missing created_at timestamps
in social media datasets, particularly for GoEmotions which lacks temporal data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def impute_missing_dates(df, method='median', reference_col='event_name', jitter_hours=6):
    """
    Impute missing created_at timestamps using various strategies.
    
    Args:
        df: DataFrame with 'created_at' column
        method: Imputation strategy:
            - 'median': Use median date from same event/source group
            - 'mean': Use mean date from same event/source group
            - 'random_range': Random date within dataset date range
            - 'sample_pool': Sample from existing dates with jitter
        reference_col: Column to group by for imputation (default: 'event_name')
        jitter_hours: Random hours to add/subtract to avoid exact duplicates (default: 6)
    
    Returns:
        DataFrame with imputed dates and tracking columns:
            - created_at: Datetime column with imputed values
            - created_at_imputed: Boolean flag indicating imputed rows
            - created_at_imputed_method: String describing imputation method used
    """
    df = df.copy()
    
    # Add imputation tracking columns
    df['created_at_imputed'] = False
    df['created_at_imputed_method'] = ''
    
    # Convert to datetime, handling empty strings and various formats
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    missing_mask = df['created_at'].isna()
    n_missing = missing_mask.sum()
    
    if n_missing == 0:
        return df
    
    print(f"Imputing {n_missing:,} missing timestamps using method: {method}")
    
    if method == 'median':
        # Use median date from same event/source group
        for group_val in df[reference_col].unique():
            group_mask = df[reference_col] == group_val
            group_dates = df.loc[group_mask, 'created_at'].dropna()
            
            if len(group_dates) > 0:
                median_date = group_dates.median()
                impute_mask = missing_mask & group_mask
                
                # Add jitter to avoid exact duplicates
                if jitter_hours > 0:
                    n_to_impute = impute_mask.sum()
                    jitter_seconds = np.random.randint(
                        -jitter_hours * 3600,
                        jitter_hours * 3600,
                        size=n_to_impute
                    )
                    imputed_dates = median_date + pd.to_timedelta(jitter_seconds, unit='s')
                    df.loc[impute_mask, 'created_at'] = imputed_dates
                else:
                    df.loc[impute_mask, 'created_at'] = median_date
                
                df.loc[impute_mask, 'created_at_imputed'] = True
                df.loc[impute_mask, 'created_at_imputed_method'] = f'median_by_{reference_col}'
    
    elif method == 'mean':
        # Use mean date from same event/source group
        for group_val in df[reference_col].unique():
            group_mask = df[reference_col] == group_val
            group_dates = df.loc[group_mask, 'created_at'].dropna()
            
            if len(group_dates) > 0:
                mean_date = group_dates.mean()
                impute_mask = missing_mask & group_mask
                
                # Add jitter to avoid exact duplicates
                if jitter_hours > 0:
                    n_to_impute = impute_mask.sum()
                    jitter_seconds = np.random.randint(
                        -jitter_hours * 3600,
                        jitter_hours * 3600,
                        size=n_to_impute
                    )
                    imputed_dates = mean_date + pd.to_timedelta(jitter_seconds, unit='s')
                    df.loc[impute_mask, 'created_at'] = imputed_dates
                else:
                    df.loc[impute_mask, 'created_at'] = mean_date
                
                df.loc[impute_mask, 'created_at_imputed'] = True
                df.loc[impute_mask, 'created_at_imputed_method'] = f'mean_by_{reference_col}'
    
    elif method == 'random_range':
        # Impute with random date within dataset range
        min_date = df['created_at'].min()
        max_date = df['created_at'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            date_range_seconds = int((max_date - min_date).total_seconds())
            random_offsets = np.random.randint(0, date_range_seconds, size=n_missing)
            random_dates = min_date + pd.to_timedelta(random_offsets, unit='s')
            
            df.loc[missing_mask, 'created_at'] = random_dates
            df.loc[missing_mask, 'created_at_imputed'] = True
            df.loc[missing_mask, 'created_at_imputed_method'] = 'random_range'
    
    elif method == 'sample_pool':
        # Sample from existing dates with jitter (best for large datasets)
        valid_dates = df.loc[~missing_mask, 'created_at'].dropna()
        
        if len(valid_dates) > 0:
            # Sample with replacement from existing dates
            sampled_dates = valid_dates.sample(n=n_missing, replace=True, random_state=42)
            
            # Add jitter to avoid exact duplicates
            if jitter_hours > 0:
                jitter_seconds = np.random.randint(
                    -jitter_hours * 3600,
                    jitter_hours * 3600,
                    size=n_missing
                )
                sampled_dates = sampled_dates.values + pd.to_timedelta(jitter_seconds, unit='s')
            
            df.loc[missing_mask, 'created_at'] = sampled_dates
            df.loc[missing_mask, 'created_at_imputed'] = True
            df.loc[missing_mask, 'created_at_imputed_method'] = f'sample_pool_jitter_{jitter_hours}h'
    
    # Fill any remaining missing values (shouldn't happen, but just in case)
    remaining_missing = df['created_at'].isna()
    if remaining_missing.any():
        # Use overall median as fallback
        fallback_date = df['created_at'].median()
        df.loc[remaining_missing, 'created_at'] = fallback_date
        df.loc[remaining_missing, 'created_at_imputed'] = True
        df.loc[remaining_missing, 'created_at_imputed_method'] = 'fallback_median'
    
    n_imputed = df['created_at_imputed'].sum()
    print(f"✓ Successfully imputed {n_imputed:,} timestamps")
    
    return df


def standardize_timestamp_format(timestamp_str):
    """
    Convert various timestamp formats to ISO 8601 standard format.
    
    Args:
        timestamp_str: String timestamp in any common format
    
    Returns:
        ISO 8601 formatted timestamp string (YYYY-MM-DDTHH:MM:SS) or None
    """
    if pd.isna(timestamp_str) or timestamp_str == '':
        return None
    
    try:
        # Use pandas flexible parsing, then convert to ISO
        dt = pd.to_datetime(timestamp_str, errors='coerce')
        if pd.notna(dt):
            return dt.isoformat()
    except:
        pass
    
    return None


def validate_dates(df, column='created_at'):
    """
    Validate date column and report statistics.
    
    Args:
        df: DataFrame with date column
        column: Name of date column to validate
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_rows': len(df),
        'valid_dates': 0,
        'missing_dates': 0,
        'imputed_dates': 0,
        'date_range': None,
        'earliest_date': None,
        'latest_date': None
    }
    
    dates = pd.to_datetime(df[column], errors='coerce')
    
    stats['valid_dates'] = dates.notna().sum()
    stats['missing_dates'] = dates.isna().sum()
    
    if f'{column}_imputed' in df.columns:
        stats['imputed_dates'] = df[f'{column}_imputed'].sum()
    
    if stats['valid_dates'] > 0:
        stats['earliest_date'] = dates.min()
        stats['latest_date'] = dates.max()
        stats['date_range'] = (stats['latest_date'] - stats['earliest_date']).days
    
    return stats
