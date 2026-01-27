# TEMPO: Twitter Emotion & Temporal Pattern Observer
## Crisis Detection through BERT-based Emotion Analysis

---

## 🎯 Project Overview

**TEMPO** analyzes Twitter data to detect fear levels during disasters and crisis events, providing early warning signals to government officials. Using BERT (Bidirectional Encoder Representations from Transformers), we classify emotions from tweets to identify when communities are experiencing heightened fear, anxiety, and distress during crises.

### Goals
1. **Emotion Detection**: Train BERT to classify 13 emotions from tweets (with focus on fear, anxiety, distress)
2. **Crisis vs. Non-Crisis Classification**: Distinguish genuine crisis emotions from excitement/entertainment
3. **Temporal Analysis**: Track emotion patterns over time to detect escalating situations
4. **Government Alerting**: Provide actionable insights for emergency response teams

### Key Use Cases
- Natural disasters (hurricanes, earthquakes, floods, wildfires)
- Public safety emergencies (shootings, bombings)
- Disease outbreaks
- Civil unrest

---

## 🔄 Data Pipeline Phases

The project follows a systematic 4-phase data preparation pipeline:

### **Phase 1: DOWNLOAD** 📥
Download raw datasets from various sources:
- **GoEmotions** (58K Reddit comments, 27 emotions) - Emotion training data
- **HumAID** (77K tweets, 19 disasters 2016-2019) - Crisis tweets with event labels
- **CrisisLex** (28K tweets, 26 events 2012-2013) - Crisis tweets with informativeness labels
- **Non-Crisis Events** (1.5M tweets) - 8 high-emotion but non-crisis events:
  - Sports: FIFA World Cup 2022, 2018, Tokyo Olympics, ICC T20
  - Entertainment: Game of Thrones, Coachella, Music Concerts  
  - Politics: US Election 2020
- **Baseline/Sentiment140** (1.6M tweets) - General Twitter noise

**Output**: Raw CSV/TSV files in respective folders

---

### **Phase 2: PROCESS** ⚙️
Clean and combine raw data:
- **Extract Timestamps**: Convert tweet IDs to timestamps using Twitter Snowflake algorithm
- **Combine HumAID Files**: Merge train/dev/test splits for each event
- **Validate Data**: Check for missing values, duplicates, encoding issues

**Output**: 
- `crisis_datasets/humaid_all_with_timestamps.csv`
- Individual event files with timestamps

**When to Re-run**: 
- After re-downloading raw data
- If timestamp extraction fails
- When adding new crisis events

---

### **Phase 3: STANDARDIZE** 🔧
Unify all datasets to common format:

**Standard Columns**:
```
text                 - Tweet text content
created_at          - Timestamp (ISO format)
event_name          - Specific event (e.g., "hurricane_harvey_2017")
event_type          - General type (e.g., "hurricane", "earthquake", "sports")
crisis_label        - Binary: 1 = crisis, 0 = non-crisis
source_dataset      - Source: "HumAID", "CrisisLex", "GoEmotions", etc.
informativeness     - How informative about the crisis (from CrisisLex)
emotion_label       - Emotion category (1-13) - TO BE ADDED
```

**Key Tasks**:
- Map specific event names → general event types
- Standardize column headers across all datasets
- Map 27 GoEmotions → 13 target emotions
- Handle missing values

**Output**:
- `standardized_data/crisis_combined.csv` (66K rows)
- `standardized_data/non_crisis_combined.csv` (1.5M rows)
- Individual standardized files

**When to Re-run**:
- After modifying event type mappings
- When emotion mapping changes (27→13)
- If new columns are added
- After fixing data quality issues

---

### **Phase 4: COMBINE** 🔗
Create final master training file:

**Process**:
1. Combine GoEmotions + Crisis + Non-Crisis data
2. Handle partial labels (not all tweets have all label types)
3. Shuffle data to prevent catastrophic forgetting
4. Create train/validation/test splits

**Output**:
- `master_training_data/master_training_data.csv` - Final training file
- `master_training_data/master_training_sample_1000.csv` - Preview sample

**Column Structure**:
```
text, emotion_label (1-13), event_type, informativeness, 
crisis_label, source_dataset, created_at
```

**When to Re-run**:
- After Phase 3 changes
- When adding new datasets
- Before training BERT model
- If shuffling seed needs to change

---

## 🔄 When to Reset/Refresh the Model

### **Complete Pipeline Re-run** (Phases 1→4)
- Adding entirely new datasets
- Major changes to emotion mapping (27→13)
- Restructuring column schema
- Starting fresh after data corruption

### **Partial Re-run** (Phase 3→4)
- Modifying event type mappings
- Fixing informativeness labels
- Adding emotion_label column
- Column name changes

### **Final Phase Only** (Phase 4)
- Changing train/val/test split ratios
- Different shuffling strategy
- Creating subsamples for testing

### **No Re-run Needed**
- Training different BERT models
- Experimenting with hyperparameters
- Creating additional analysis notebooks

---

## 📊 Current Data Status

**Available** ✅:
- Crisis combined: 66,748 tweets
- Non-crisis combined: 1,533,696 tweets
- Standardized datasets with columns: `text, created_at, event_name, event_type, crisis_label, source_dataset, informativeness`

**Missing** ❌:
- GoEmotions data (27 emotions)
- Emotion labels (need to add `emotion_label` column)
- Master training file

**Next Priority**:
1. Download GoEmotions from Google Drive
2. Define 13 target emotions
3. Create 27→13 emotion mapping
4. Add emotion_label column to datasets

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.14+
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy jupyter ipykernel
```

### Setup
1. Clone this repository
2. Download data folders from Google Drive to project root:
   - `standardized_data/`
   - `goemotion_data/`
   - `master_training_data/` (optional)
3. Run inspection notebooks:
   - `00_project_status.ipynb` - Overview of all data
   - `01_data_inspection.ipynb` - Detailed analysis

### Repository Structure
```
TEMPO/
├── README.md                          # This file
├── SCRIPTS_REFERENCE.md               # Documentation of all scripts/notebooks
├── .gitignore                         # Excludes data folders
│
├── scripts/                           # Data pipeline scripts
│   ├── phase1_download/               # Download from sources
│   ├── phase2_process/                # Clean and combine
│   ├── phase3_standardize/            # Unify format
│   └── phase4_combine/                # Create master file
│
├── utils/                             # Analysis and inspection tools
│   ├── check_goemotions_baseline.py
│   ├── check_crisis_events.py
│   ├── explore_non_crisis.py
│   └── inspect_large_dataset.py
│
├── 00_project_status.ipynb            # Dashboard of all data files
├── 01_data_inspection.ipynb           # Detailed data inspection
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

## 📝 Documentation

- See [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md) for detailed documentation of each script and notebook

---

## 👥 Team

JIKI DAP Round 2 Team
January 2026

---

## 📄 License

[Add license information]
