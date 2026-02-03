"""Tests for created_at imputation: pool sampling, deterministic fallback, jitter bounds, and audit columns."""

import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from importlib.util import spec_from_file_location, module_from_spec


def load_module(path):
    spec = spec_from_file_location('create_master', str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pool_sampling_imputes_and_sets_audit(tmp_path):
    # Create a small sample DataFrame with missing created_at
    df = pd.DataFrame({
        'text': ['a', 'b', 'c', 'd'],
        'emotion_fear': [0, 0, 0, 0],
        'event_type': [None, 'hurricane', None, 'flood'],
        'informativeness': [None, None, None, None],
        'crisis_label': [None, 1, None, 1],
        'source_dataset': ['goemotions', 'humaid', 'goemotions', 'crisislex'],
        'created_at': [None, '2018-07-01 12:00:00', None, '2019-05-01 13:00:00']
    })

    # Create small pool files
    pool1 = tmp_path / 'pool1.csv'
    pool2 = tmp_path / 'pool2.csv'
    pd.DataFrame({'created_at': ['2018-07-01 10:00:00', '2018-07-01 11:00:00']}).to_csv(pool1, index=False)
    pd.DataFrame({'created_at': ['2019-05-01 11:00:00']}).to_csv(pool2, index=False)

    # Load module and call impute
    mod = load_module(Path('scripts/phase4_combine/create_master_training_file.py'))
    imputed = mod.impute_created_at(df, pool_paths=[str(pool1), str(pool2)], seed=123, jitter_hours=1)

    assert 'created_at_imputed' in imputed.columns
    assert 'created_at_imputed_method' in imputed.columns

    # Two missing entries should be imputed
    assert int(imputed['created_at_imputed'].sum()) == 2

    # Methods should indicate pool sampling
    methods = imputed.loc[imputed['created_at_imputed'], 'created_at_imputed_method'].unique()
    assert all(m == 'sampling_pool_jitter' for m in methods)

    # All created_at parsed correctly
    parsed = pd.to_datetime(imputed['created_at'], errors='coerce')
    assert parsed.notna().all()


def test_fixed_fallback_when_pool_empty(tmp_path):
    df = pd.DataFrame({
        'text': ['x', 'y'],
        'created_at': [None, None]
    })

    mod = load_module(Path('scripts/phase4_combine/create_master_training_file.py'))
    imputed = mod.impute_created_at(df, pool_paths=[], seed=42, jitter_hours=6)

    assert int(imputed['created_at_imputed'].sum()) == 2
    assert (imputed['created_at_imputed_method'] == 'fixed_fallback').all()

    # Check that fallback timestamp matches expected formatted string
    assert (imputed['created_at'] == '2018-06-30 23:53:08').all()


def test_jitter_within_bounds(tmp_path):
    df = pd.DataFrame({
        'text': ['m', 'n', 'o'],
        'created_at': [None, None, None]
    })

    pool = tmp_path / 'pool.csv'
    pd.DataFrame({'created_at': ['2020-01-01 12:00:00']}).to_csv(pool, index=False)

    mod = load_module(Path('scripts/phase4_combine/create_master_training_file.py'))

    # jitter_hours=2 means +/- 2 hours
    imputed = mod.impute_created_at(df, pool_paths=[str(pool)], seed=99, jitter_hours=2)
    times = pd.to_datetime(imputed['created_at'])

    base = pd.to_datetime('2020-01-01 12:00:00')
    diffs = (times - base).abs()
    # All diffs should be <= 2 hours and > 0 (since jitter expected)
    assert (diffs <= pd.Timedelta(hours=2)).all()


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
