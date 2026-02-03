"""
Impute missing created_at in master sample and write a new file.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SAMPLE_IN = Path('master_training_data/master_training_sample_10kv3.csv')
SAMPLE_OUT = Path('master_training_data/master_training_sample_10kv3_imputed.csv')
POOL_FILES = [Path('standardized_data/crisis_combined_dates_only.csv'), Path('standardized_data/non_crisis_combined.csv')]
SEED = 42
JITTER_HOURS = 6

def main():
    print('Loading sample...')
    df = pd.read_csv(SAMPLE_IN, low_memory=False)
    print('Sample rows:', len(df))
    df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
    missing_mask = df['created_at_parsed'].isna()
    print('Missing before:', missing_mask.sum())

    pool_ts = []
    for f in POOL_FILES:
        if f.exists():
            tmp = pd.read_csv(f, usecols=['created_at'], low_memory=False)
            pool_ts.extend(pd.to_datetime(tmp['created_at'], errors='coerce').dropna().tolist())
    print('Pool size:', len(pool_ts))

    rng = np.random.default_rng(SEED)

    if 'created_at_imputed' not in df.columns:
        df['created_at_imputed'] = False
    if 'created_at_imputed_method' not in df.columns:
        df['created_at_imputed_method'] = None

    if len(pool_ts)==0:
        fallback = pd.to_datetime('2018-06-30 23:53:08')
        for idx in df[missing_mask].index:
            df.at[idx,'created_at_parsed'] = fallback
            df.at[idx,'created_at_imputed'] = True
            df.at[idx,'created_at_imputed_method'] = 'fixed_fallback'
    else:
        pool_arr = np.array(pool_ts, dtype='datetime64[ns]')
        miss_idx = df[missing_mask].index.to_numpy()
        choices = rng.integers(0, len(pool_arr), size=len(miss_idx))
        jitters = rng.integers(-JITTER_HOURS*3600, JITTER_HOURS*3600, size=len(miss_idx))
        for i, idx in enumerate(miss_idx):
            sampled = pd.to_datetime(pool_arr[choices[i]])
            dt = sampled + pd.Timedelta(seconds=int(jitters[i]))
            df.at[idx,'created_at_parsed'] = dt
            df.at[idx,'created_at_imputed'] = True
            df.at[idx,'created_at_imputed_method'] = 'sampling_pool_jitter'

    print('Imputed count:', int(df['created_at_imputed'].sum()))
    df['created_at'] = df['created_at_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['created_at_parsed'])
    print('Writing to', SAMPLE_OUT)
    df.to_csv(SAMPLE_OUT, index=False)
    print('Done.')

if __name__ == '__main__':
    main()
