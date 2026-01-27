# TEMPO Scripts & Notebooks Reference
## Complete Documentation of All Code Files

---

## 📓 Notebooks

### **00_project_status.ipynb** 📊
**Purpose**: Quick dashboard to visualize current state of all data files

**What it shows**:
- Which data folders exist vs. missing
- All CSV files with sizes
- Dataset row counts and column structures
- Issues and missing components
- Quick statistics by event type

**When to use**:
- First time setup to verify data download
- After downloading new data from Google Drive
- To get quick overview without loading large files
- Before running pipeline to check prerequisites

**Runtime**: ~5-10 seconds (optimized with Unix commands)

---

### **01_data_inspection.ipynb** 🔍
**Purpose**: Detailed analysis of standardized datasets

**What it shows**:
- Complete column structure for crisis and non-crisis data
- Informativeness value distribution
- Event type mappings
- Emotion column status (currently none - to be added)
- Data types and missing values

**When to use**:
- After Phase 3 (Standardize) completion
- To understand data structure before Phase 4
- When debugging data quality issues
- Before creating emotion mapping

**Runtime**: ~5-15 seconds (loads full datasets)

---

## 📁 Phase 1: Download Scripts

### **download_baseline.py**
**Purpose**: Downloads Sentiment140 dataset from Kaggle (1.6M general tweets)

**What it does**:
- Downloads ~238MB zip file
- Extracts to `baseline_data/`
- Provides baseline "noise" tweets unrelated to events

**Prerequisites**:
- Kaggle API configured (`~/.kaggle/kaggle.json`)
- `kaggle` package installed

**Output**: `baseline_data/training.1600000.processed.noemoticon.csv`

**Note**: File has NO header row. Column order: `sentiment, id, date, query, user, text`

---

### **download_crisislex.py**
**Purpose**: Downloads CrisisLex T26 dataset from GitHub (28K crisis tweets, 26 events 2012-2013)

**What it does**:
- Downloads labeled tweets AND timestamp files
- Merges them together
- Saves individual event CSVs + combined file

**Output**:
- `crisis_datasets/crisislex_complete/` (individual events)
- `crisis_datasets/crisislex_all_complete.csv` (combined)

**Key columns**: `Tweet Text, Informativeness, event_name, created_at`

**Events included**: 2012 Colorado Wildfires, Hurricane Sandy, 2013 Boston Bombings, Oklahoma Tornadoes, various earthquakes/floods

---

### **download_goemotions.py**
**Purpose**: Downloads GoEmotions dataset from Hugging Face (58K Reddit comments, 27 emotions)

**What it does**:
- Downloads from `google-research-datasets/go_emotions`
- Combines train/validation/test splits
- Provides emotion training data

**Output**: `goemotion_data/goemotions.csv`

**Key use**: Source of 27 emotions that will be mapped to 13 target emotions

**Prerequisites**: `pip install datasets pandas`

---

### **download_humaid.py**
**Purpose**: Downloads HumAID dataset from CrisisNLP (77K tweets, 19 disasters 2016-2019)

**What it does**:
- Downloads Set 1 (47K tweets) from CrisisNLP
- Extracts tar.gz archive
- Saves multiple TSV files per event

**Output**: `crisis_datasets/humaid_crisis_data/*.tsv`

**Events**: Hurricanes (Harvey, Irma, Florence, Matthew), Wildfires (California, Canada, Greece), Earthquakes (Nepal, Mexico, Kaikoura), Floods (Kerala, Sri Lanka, Maryland)

**Note**: For Set 2 (29K more tweets), fill out form at CrisisNLP website

---

### **download_non_crisis.py**
**Purpose**: Downloads 8 non-crisis high-emotion datasets from Kaggle

**What it does**:
- Downloads sports events (FIFA World Cup 2022, 2018, Tokyo Olympics, ICC T20)
- Downloads entertainment (Game of Thrones, Coachella, Music Concerts)
- Downloads politics (US Election 2020)
- Helps model distinguish crisis from excitement/entertainment

**Output**: `non_crisis_data/[dataset_folders]/*.csv`

**Prerequisites**: Kaggle API configured

**Purpose**: Provides contrast - these generate high Twitter activity but are NOT crises

---

## 📁 Phase 2: Process Scripts

### **extract_humaid_timestamps.py**
**Purpose**: Converts HumAID tweet IDs to timestamps using Twitter Snowflake algorithm

**What it does**:
- Reads TSV files from `crisis_datasets/humaid_crisis_data/`
- Extracts timestamp from tweet ID (bits 63-22)
- Twitter epoch: 2010-11-04 01:42:54 UTC
- Adds `created_at` column

**Output**: `crisis_datasets/humaid_crisis_data/*_with_timestamps.csv`

**Why needed**: Essential for temporal analysis (1hr ago, 3hr ago features for RL)

**Formula**: `timestamp_ms = (tweet_id >> 22) + 1288834974657`

---

### **combine_humaid_files.py**
**Purpose**: Merges all HumAID event files into single master file

**What it does**:
- Finds all `*_with_timestamps.csv` files
- Combines train/dev/test splits
- Adds `event_name`, `source_dataset`, `crisis_label` metadata

**Output**: `crisis_datasets/humaid_all_with_timestamps.csv`

**Prerequisites**: Run `extract_humaid_timestamps.py` first

**Shows**: Event breakdown, date ranges, total tweets per event

---

## 📁 Phase 3: Standardize Scripts

### **standardize_crisis_data.py**
**Purpose**: Unifies HumAID and CrisisLex to common format and maps event types

**What it does**:
- Reads `humaid_all_with_timestamps.csv` and `crisislex_all_complete.csv`
- Maps specific event names → general types (e.g., "hurricane_harvey" → "hurricane")
- Standardizes column headers
- Handles informativeness labels

**Output**:
- `standardized_data/humaid_standardized.csv`
- `standardized_data/crisislex_standardized.csv`
- `standardized_data/crisis_combined.csv`

**Standard columns**: `text, created_at, event_name, event_type, crisis_label, source_dataset, informativeness`

**Event type mapping**:
- Weather: hurricane, flood, tornado, wildfire, haze
- Geological: earthquake, tsunami, landslide
- Violence: shooting, bombing, attack
- Civil unrest: protest
- Health: disease_outbreak
- Other: accident, drought, heatwave, sinkhole

---

### **standardize_non_crisis_data.py**
**Purpose**: Unifies all 8 non-crisis datasets to common format

**What it does**:
- Processes each of 8 datasets (different column names per source)
- Extracts text and timestamp columns
- Assigns `event_name` and `event_type` labels
- Sets `crisis_label = 0` for all

**Output**:
- Individual: `standardized_data/[dataset]_standardized.csv`
- Combined: `standardized_data/non_crisis_combined.csv`

**Standard columns**: `text, created_at, event_name, event_type, crisis_label, source_dataset`

**Event types**: `sports`, `entertainment`, `politics`

**Note**: Tweet IDs NOT included due to Excel precision issues (fine for BERT training)

---

## 📁 Phase 4: Combine Scripts

### **create_master_training_file.py**
**Purpose**: Creates final master training file combining all datasets

**What it does**:
- Combines GoEmotions + Crisis + Non-Crisis
- Handles partial labels (not all tweets have all 3 label types)
- Shuffles data to prevent catastrophic forgetting
- Creates train/val/test splits

**Input**:
- `goemotion_data/goemotions.csv`
- `standardized_data/crisis_combined.csv`
- `standardized_data/non_crisis_combined.csv`

**Output**:
- `master_training_data/master_training_data.csv`
- `master_training_data/master_training_sample_1000.csv` (preview)

**Column structure**:
```
text, emotion_fear, emotion_anger, emotion_joy, ... (13 emotions),
event_type, informativeness, crisis_label, source_dataset, created_at
```

**Partial labels handling**:
- GoEmotions: Has emotions, NULL event_type/informativeness
- Crisis: Has event_type/informativeness, NULL emotions
- Non-Crisis: Has event_type, NULL emotions/informativeness

**Note**: Baseline dataset NOT included (non-crisis provides sufficient contrast)

---

## 🛠️ Utility Scripts

### **check_goemotions_baseline.py**
**Purpose**: Pre-check GoEmotions format and see ALL 28 emotions

**What it shows**:
- All 28 GoEmotions emotion labels (0-27)
- Emotion distribution
- Baseline dataset format (if present)
- Recommendations for which 13 emotions to use

**When to use**:
- BEFORE running `create_master_training_file.py`
- To decide which 13 emotions to target
- To validate GoEmotions download

**Output**: Console output with emotion analysis

---

### **check_crisis_events.py**
**Purpose**: Validates HumAID and CrisisLex event mappings

**What it shows**:
- All unique events in crisis datasets
- Which events map to which event types
- Unmapped events that need mapping
- Validates all important columns present

**When to use**:
- BEFORE running `standardize_crisis_data.py`
- After adding new crisis events
- To verify event type mappings are complete

**Output**: Console output showing event mappings and recommendations

---

### **explore_non_crisis.py**
**Purpose**: Discovers and analyzes all non-crisis CSV files

**What it shows**:
- All CSV files in `non_crisis_data/`
- Column names (vary between datasets)
- Text, timestamp, and ID columns
- Which datasets need timestamp extraction

**When to use**:
- BEFORE running `standardize_non_crisis_data.py`
- After downloading new non-crisis datasets
- To identify column names for standardization

**Output**: Console output + optional `non_crisis_summary.csv`

---

### **inspect_large_dataset.py**
**Purpose**: Inspects CSV files too large for Excel (>1M rows)

**What it shows**:
- File size in MB
- Total row count (memory-efficient)
- Sample data
- Full distribution analysis using chunking

**When to use**:
- To view `non_crisis_combined.csv` (1.5M rows)
- To view `master_training_data.csv` (can be very large)
- When Excel can't open file

**Configuration**: Edit `file_path` variable in script

**Default**: `standardized_data/non_crisis_combined.csv`

---

## 🔄 Execution Order

### **Fresh Start (Download All Data)**
```
Phase 1: Download
├─ download_goemotions.py
├─ download_humaid.py
├─ download_crisislex.py
├─ download_non_crisis.py
└─ download_baseline.py (optional)

Phase 2: Process
├─ extract_humaid_timestamps.py
└─ combine_humaid_files.py

Phase 3: Standardize
├─ check_crisis_events.py (validate first)
├─ standardize_crisis_data.py
├─ explore_non_crisis.py (validate first)
└─ standardize_non_crisis_data.py

Phase 4: Combine
├─ check_goemotions_baseline.py (validate first)
└─ create_master_training_file.py
```

### **Using Pre-Downloaded Data from Google Drive**
```
1. Download folders to project root:
   - standardized_data/
   - goemotion_data/
   - (optionally) master_training_data/

2. Run inspection:
   - 00_project_status.ipynb
   - 01_data_inspection.ipynb

3. If needed, re-run phases 3-4:
   - standardize_crisis_data.py
   - standardize_non_crisis_data.py
   - create_master_training_file.py
```

---

## 📝 Quick Command Reference

```bash
# Phase 1 - Download
python scripts/phase1_download/download_goemotions.py
python scripts/phase1_download/download_humaid.py
python scripts/phase1_download/download_crisislex.py
python scripts/phase1_download/download_non_crisis.py

# Phase 2 - Process
python scripts/phase2_process/extract_humaid_timestamps.py
python scripts/phase2_process/combine_humaid_files.py

# Phase 3 - Standardize
python utils/check_crisis_events.py  # Check first
python scripts/phase3_standardize/standardize_crisis_data.py
python utils/explore_non_crisis.py   # Check first
python scripts/phase3_standardize/standardize_non_crisis_data.py

# Phase 4 - Combine
python utils/check_goemotions_baseline.py  # Check first
python scripts/phase4_combine/create_master_training_file.py

# Utilities
python utils/inspect_large_dataset.py
```

---

## 🎯 Common Tasks

### "I just downloaded data from Google Drive"
```
1. Run: 00_project_status.ipynb (verify all folders present)
2. Run: 01_data_inspection.ipynb (understand structure)
3. Check what's missing
4. Proceed with next steps
```

### "I need to change event type mappings"
```
1. Edit: scripts/phase3_standardize/standardize_crisis_data.py
2. Run: python scripts/phase3_standardize/standardize_crisis_data.py
3. Run: python scripts/phase4_combine/create_master_training_file.py
```

### "I need to map 27→13 emotions"
```
1. Run: python utils/check_goemotions_baseline.py (see all 27)
2. Define your 13 target emotions
3. Create mapping dictionary
4. Edit: scripts/phase4_combine/create_master_training_file.py
5. Re-run phase 4
```

### "I want to inspect a large CSV file"
```
1. Edit: utils/inspect_large_dataset.py (set file_path)
2. Run: python utils/inspect_large_dataset.py
```

---

## ⚙️ Configuration

Most scripts use hardcoded paths relative to project root:
- Input: `./crisis_datasets/`, `./goemotion_data/`, etc.
- Output: `./standardized_data/`, `./master_training_data/`, etc.

To change paths, edit the `CONFIGURATION` section in each script.

---

**Last Updated**: January 2026  
**Team**: JIKI DAP Round 2
