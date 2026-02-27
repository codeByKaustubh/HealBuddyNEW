# HealBuddy Symptom Checker

A Streamlit-based symptom checker demo with three ML models and local explainability using SHAP and LIME.

## Features
- Text-first symptom input with synonym handling (for example `stomach ache -> Abdominal Pain`)
- Three model options:
  - Random Forest
  - Extra Trees
  - K-Nearest Neighbors
- Top predictions with probability thresholding
- SHAP and LIME local explanations
- Leakage-safe model evaluation metrics in UI:
  - Holdout Accuracy
  - Holdout Macro F1
  - Cross-validation Macro F1 (grouped by symptom pattern)

## Project Structure
- `app.py`: Streamlit UI and prediction flow
- `evaluate_models.py`: CLI model evaluation
- `src/config.py`: project constants and tunables
- `src/data.py`: data loading and symptom parsing
- `src/models.py`: model training and evaluation
- `src/explainability.py`: SHAP/LIME helpers

## Quick Start
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

3. Run evaluation in terminal:

```bash
python evaluate_models.py
```

4. Clean and audit dataset (dedup + conflict report):

```bash
python clean_dataset.py --input symptom_disease_dataset_realistic_duplicated.csv --output cleaned_dataset.csv --report quality_report.json --conflicts conflict_patterns.csv --strategy drop_conflicts
```

## Notes
- This project is for educational/demo use only and is not a medical diagnosis system.
- The default dataset includes many duplicated rows; evaluation is leakage-safe by grouping identical symptom patterns during holdout/CV.
