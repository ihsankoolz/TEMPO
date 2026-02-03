# TEMPO: Twitter Emotion & Temporal Pattern Observer
## Crisis Detection using NLP + Reinforcement Learning

---

## Project Overview

**TEMPO** combines NLP (BERT for emotion extraction) with Reinforcement Learning (RL) for optimal crisis alert timing. The system learns to distinguish genuine crises from high-emotion non-crisis events (sports, entertainment) and determines the optimal moment to alert emergency responders.

### Goals
1. **Emotion Detection**: Train multi-task BERT to classify 13 emotions from tweets
2. **Crisis vs. Non-Crisis Classification**: Distinguish genuine crisis emotions from excitement/entertainment
3. **Temporal Analysis**: Track emotion patterns over time using multi-timescale features
4. **RL Alert Timing**: Learn optimal alert timing to maximize detection while minimizing false alarms

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│  Master Training Data (~217K rows)                          │
│  - GoEmotions (54K, labeled emotions)                       │
│  - Crisis (67K) + Non-crisis (96K sampled)                  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
              Train Multi-task BERT
              (Emotion + Crisis classification)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Apply BERT to ORIGINAL FULL data                           │
│  - Crisis: 67K tweets                                       │
│  - Non-crisis: 1.5M+ tweets                                 │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
         Extract emotion features per tweet
                      │
                      ▼
         Create episodes & hourly aggregations
                      │
                      ▼
              Train RL Agent (alert timing)
```

---

## Data Pipeline Phases

### **Phase 1: DOWNLOAD**
Download raw datasets from various sources:
- **GoEmotions** (54K Reddit comments, 27 emotions) - Emotion training data
- **HumAID** (43K tweets, 13 disasters) - Crisis tweets with event labels
- **CrisisLex** (23K tweets, 16 events) - Crisis tweets with informativeness labels
- **Non-Crisis Events** (1.5M tweets):
  - Sports: FIFA World Cup 2018/2022, Tokyo Olympics
  - Entertainment: Game of Thrones, Coachella, Music Concerts
  - Politics: US Election 2020
- **Baseline/Sentiment140** - General Twitter noise

---

### **Phase 2: PROCESS**
Clean and combine raw data:
- Extract timestamps from tweet IDs (Twitter Snowflake algorithm)
- Combine HumAID train/dev/test splits
- Validate data quality

---

### **Phase 3: STANDARDIZE**
Unify all datasets to common format:

**Standard Columns**:
| Column | Description |
|--------|-------------|
| `text` | Tweet/comment text content |
| `created_at` | Timestamp (ISO format) |
| `event_name` | Specific event (e.g., "hurricane_harvey_2017") |
| `event_type` | General type (e.g., "hurricane", "sports") |
| `crisis_label` | Binary: 1 = crisis, 0 = non-crisis |
| `source_dataset` | Origin: "humaid", "crisislex", "GoEmotions", etc. |
| `informativeness` | CrisisLex label (see below) |
| `emotion_label` | Numeric emotion (1-13) |
| `emotion_name` | Text emotion name |

**Informativeness Labels** (CrisisLex data only):
| Label | Count | Description |
|-------|-------|-------------|
| `related_informative` | 14,379 | About crisis AND contains actionable info |
| `related_not_informative` | 6,257 | About crisis but no useful details |
| `not_related` | 2,296 | Noise/unrelated tweets |
| `NaN` | 43,816 | HumAID data (no informativeness rating) |

**Output**:
- `standardized_data/crisis_combined.csv` (66,748 rows)
- `standardized_data/non_crisis_combined.csv` (1,533,696 rows)

---

### **Phase 4: COMBINE**
Create master training file for BERT:

**Sampling Strategy** (to balance dataset):
- **GoEmotions**: 54,263 (all - has emotion labels)
- **Crisis**: 66,748 (all - core crisis data)
- **Non-crisis**: ~96K (sampled from 1.5M with sports emphasis)

**Non-crisis Sampling Distribution**:
| Source | Type | Sample Size |
|--------|------|-------------|
| worldcup_2018 | Sports | 20,000 |
| tokyo_olympics | Sports | 20,000 |
| fifa_worldcup | Sports | 20,000 |
| game_of_thrones | Entertainment | 20,000 |
| us_election | Politics | 10,000 |
| coachella | Entertainment | ~3,846 (all) |
| music_concerts | Entertainment | ~1,830 (all) |

**Output**:
- `master_training_data/master_training_data_v3.csv` (~217K rows, shuffled)
- `master_training_data/master_training_sample_10k.csv` (for preview)

---

## 13 Target Emotions

Mapped from 27 GoEmotions to 13 crisis-optimized categories:

| Label | Emotion | Mapping Source |
|-------|---------|----------------|
| 1 | fear | fear |
| 2 | anger | anger + annoyance |
| 3 | sadness | sadness + grief |
| 4 | anxiety | nervousness |
| 5 | confusion | confusion + curiosity |
| 6 | surprise | surprise + realization |
| 7 | disgust | disgust + embarrassment |
| 8 | caring | caring |
| 9 | joy | joy + amusement + love + admiration |
| 10 | excitement | excitement + desire |
| 11 | gratitude | gratitude + optimism + pride + relief |
| 12 | disappointment | disappointment + disapproval + remorse |
| 13 | neutral | neutral + approval |

---

## Current Data Status

**Completed**:
- Crisis combined: 66,748 tweets
- Non-crisis combined: 1,533,696 tweets
- GoEmotions with 13 emotions: 54,263 rows
- Emotion mapping config: `emotion_mapping_config.json`
- Master training file: `master_training_data_v3.csv` (~217K rows)

**Next Steps**:
1. Train multi-task BERT on master training data
2. Apply BERT to full crisis/non-crisis datasets
3. Create episodes with hourly aggregations
4. Train RL agent for alert timing

---

## Repository Structure
```
TEMPO/
├── README.md
├── emotion_mapping_config.json        # 27→13 emotion mapping
│
├── scripts/
│   ├── phase1_download/               # Download from sources
│   ├── phase2_process/                # Clean and combine
│   ├── phase3_standardize/            # Unify format
│   └── phase4_combine/                # Create master file
│
├── utils/                             # Analysis tools
│
├── 00_project_status.ipynb            # Dashboard
├── 01_data_inspection.ipynb           # Data inspection
├── 02_emotion_mapping.ipynb           # Create 27→13 mapping
├── 03_apply_emotion_mapping.ipynb     # Apply mapping to GoEmotions
├── 04_add_emotion_columns_to_standardized_data.ipynb
├── 05_phase4_create_master_training_file.ipynb
│
└── [data folders - not in git]
    ├── standardized_data/             # Processed data
    ├── crisis_datasets/               # Raw crisis data
    ├── goemotion_data/                # Emotion labels
    ├── non_crisis_data/               # Non-crisis events
    ├── master_training_data/          # Final training file
    └── baseline_data/                 # General tweets
```

---

## Team

JIKI DAP Round 2 Team - January 2026
- Iain Nabiel (Y1 CS)
- Muhammad Khaleil (Y1 IS)
- Lim Junsheng (Y2 CS)
- Muhammad Ihsan Bin Alfian (Y2 IS)

---

## References

Benhamou, E., Saltiel, D., Ohana, J. J., & Atif, J. (2021). Detecting and adapting to crisis pattern with context based Deep Reinforcement Learning. In *2020 25th International Conference on Pattern Recognition (ICPR)* (pp. 10050-10057). IEEE.
