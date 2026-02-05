# TEMPO Setup Instructions

## 1. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Setup Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key at: https://makersuite.google.com/app/apikey

## 4. Select Kernel in VS Code

1. Open any notebook (e.g., `01_data_inspection.ipynb`)
2. Click "Select Kernel" in top-right
3. Choose "Python Environments"
4. Select `venv` (Python 3.14.0)

## 5. Run Notebooks in Order

1. `00_project_status.ipynb` - Project overview
2. `01_data_inspection.ipynb` - Inspect data
3. `02_emotion_mapping.ipynb` - Define emotion mapping
4. `03_apply_emotion_mapping.ipynb` - Apply mapping to GoEmotions
5. `04_add_emotion_columns_to_standardized_data.ipynb` - Add columns
6. `05_phase4_create_master_training_file.ipynb` - Create master dataset
7. `06_llm_label_missing_data.ipynb` - Fill missing labels with LLM

## Troubleshooting

**ModuleNotFoundError**: Make sure you activated the venv and installed requirements.txt

**API Key Error**: Check your `.env` file has the correct API key

**Kernel Issues**: Restart VS Code or run "Python: Select Interpreter" command
