# JIKI: Twitter Crisis Detection using NLP + Reinforcement Learning

## Project Overview

This project builds an end-to-end machine learning system that combines NLP (Natural Language Processing) and Reinforcement Learning for Twitter crisis detection. The system learns optimal alert timing for crisis events by analyzing emotion patterns, event types, and temporal features from Twitter data.

**Key Innovation**: Instead of static rule-based thresholds, our RL agent learns context-aware decisions about WHEN to alert during evolving crisis situations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Task BERT Model                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │  Emotion    │  │  Event Type │  │  Informativeness    │     │
│  │  Extractor  │  │  Classifier │  │  Classifier         │     │
│  │ (13 emotions)│  │(crisis/non) │  │  (CrisisLex)        │     │
│  └─────────────┘  └─────────────┘  └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RL Alert Timing Agent                        │
│  State: [emotions_multi_timescale, event_type, volatility]     │
│  Actions: [alert, wait_1hr, wait_2hr, dismiss]                 │
│  Reward: +15 correct alert, -5 false alarm, -20 missed crisis  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Pipeline

```
Phase 1: DOWNLOAD          Phase 2: PROCESS           Phase 3: STANDARDIZE       Phase 4: COMBINE
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ GoEmotions       │      │ Extract          │      │ Standardize      │      │ Master Training  │
│ HumAID           │ ───► │ Timestamps       │ ───► │ Column Headers   │ ───► │ File             │
│ CrisisLex        │      │ Combine Files    │      │ Map Event Types  │      │ (Shuffled)       │
│ Non-Crisis (8)   │      │                  │      │                  │      │                  │
│ Baseline         │      │                  │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Folder Structure

```
JIKI/
├── README.md                           # This file
├── scripts/
│   ├── phase1_download/                # Data collection scripts
│   │   ├── download_goemotions.py
│   │   ├── download_baseline.py
│   │   ├── download_humaid.py
│   │   ├── download_crisislex.py
│   │   └── download_non_crisis.py
│   ├── phase2_process/                 # Data processing scripts
│   │   ├── extract_humaid_timestamps.py
│   │   └── combine_humaid_files.py
│   ├── phase3_standardize/             # Standardization scripts
│   │   ├── standardize_crisis_data.py
│   │   └── standardize_non_crisis_data.py
│   └── phase4_combine/                 # Final combination
│       └── create_master_training_file.py
├── utils/                              # Helper/utility scripts
│   ├── check_crisis_events.py
│   ├── check_goemotions_baseline.py
│   ├── explore_non_crisis.py
│   └── inspect_large_dataset.py
├── archive/                            # Obsolete/draft scripts
│   ├── download_crisislex_old.py
│   └── test_hurricane_harvey.py
├── data/                               # Raw downloaded data
│   ├── goemotion_data/
│   ├── baseline_data/
│   ├── crisis_datasets/
│   └── non_crisis_data/
├── standardized_data/                  # Processed standardized data
│   ├── crisis_combined.csv
│   ├── non_crisis_combined.csv
│   └── [individual standardized files]
└── master_training_data/               # Final training data
    ├── master_training_data.csv
    └── master_training_sample_1000.csv
```

## Datasets

### Training Data Sources

| Dataset | Purpose | Rows | Labels |
|---------|---------|------|--------|
| GoEmotions | Emotion classification | ~58K | 13 emotions (fear, anger, joy, etc.) |
| HumAID | Crisis events | ~77K | Event type, timestamps |
| CrisisLex T26 | Crisis events | ~28K | Event type, informativeness, timestamps |
| Non-Crisis (8 datasets) | Non-crisis events | ~2M+ | Event type (sports, entertainment, politics) |
| Sentiment140 | Baseline noise | 1.6M | None (general noise) |

### Non-Crisis Datasets
- FIFA World Cup 2022
- FIFA World Cup 2018
- Tokyo Olympics 2020
- US Election 2020
- Game of Thrones Season 8
- Coachella 2015
- Music Concerts 2021
- ICC T20 World Cup 2021

### Standardized Column Format

All datasets are standardized to:
```
text | created_at | event_name | event_type | crisis_label | source_dataset | informativeness
```

| Column | Description |
|--------|-------------|
| `text` | Tweet content |
| `created_at` | UTC timestamp |
| `event_name` | Specific event (e.g., `hurricane_harvey_2017`) |
| `event_type` | Category (e.g., `hurricane`, `sports`, `entertainment`) |
| `crisis_label` | 1 = crisis, 0 = non-crisis |
| `source_dataset` | Origin dataset name |
| `informativeness` | CrisisLex label (if available) |

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy datasets kaggle transformers torch
```

### 2. Configure Kaggle API (for downloading datasets)
```bash
# Get API key from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run Data Pipeline

```bash
# Phase 1: Download all datasets
python scripts/phase1_download/download_goemotions.py
python scripts/phase1_download/download_humaid.py
python scripts/phase1_download/download_crisislex.py
python scripts/phase1_download/download_non_crisis.py
python scripts/phase1_download/download_baseline.py

# Phase 2: Process (extract timestamps, combine files)
python scripts/phase2_process/extract_humaid_timestamps.py
python scripts/phase2_process/combine_humaid_files.py

# Phase 3: Standardize
python scripts/phase3_standardize/standardize_crisis_data.py
python scripts/phase3_standardize/standardize_non_crisis_data.py

# Phase 4: Create master training file
python scripts/phase4_combine/create_master_training_file.py
```

### 4. Output
After running the pipeline, you'll have:
- `master_training_data/master_training_data.csv` - Ready for multi-task BERT training

## Event Type Mapping

### Crisis Events
| Event Name Pattern | Event Type |
|-------------------|------------|
| hurricane_*, cyclone_*, typhoon_* | `hurricane` |
| earthquake_*, quake_* | `earthquake` |
| flood_*, flooding_* | `flood` |
| wildfire_*, fire_*, bushfire_* | `wildfire` |
| shooting_*, gunfire_* | `shooting` |
| bombing_*, explosion_* | `bombing` |
| covid_*, pandemic_* | `disease_outbreak` |

### Non-Crisis Events
| Dataset | Event Type |
|---------|------------|
| FIFA World Cup, Olympics, ICC T20 | `sports` |
| Coachella, Music Concerts, Game of Thrones | `entertainment` |
| US Election | `politics` |

## Key Features

### Multi-Task Learning
- **One BERT model** with 3 output heads (prevents catastrophic forgetting)
- Handles **partial labels** (not all tweets have all 3 label types)
- **13 emotions** for richer crisis detection (not just fear/anger/joy)

### Temporal Features (for RL)
- Multi-timescale emotions (current, 1hr ago, 3hr ago, 6hr ago)
- Emotion volatility (standard deviation over 3-hour window)
- Tweet volume tracking

### Episode Structure
```python
episode = {
    'event_id': 'hurricane_harvey_2017',
    'is_crisis': True,
    'optimal_alert_hour': 3,
    'timeline': [
        {'hour': 0, 'state': [...], 'action': 'wait', 'reward': 0},
        {'hour': 1, 'state': [...], 'action': 'wait', 'reward': 0},
        {'hour': 2, 'state': [...], 'action': 'alert', 'reward': +15},
    ]
}
```



## References

- Benhamou, E., et al. (2021). "Detecting and adapting to crisis pattern with context based Deep Reinforcement Learning." ICPR 2020.
- GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
- HumAID Dataset: https://crisisnlp.qcri.org/humaid_dataset
- CrisisLex: https://crisislex.org/

## License

This project is for educational purposes as part of the DAP (Data Analytics Project) program.
