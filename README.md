# HealBuddy Symptom Checker

A Streamlit-based multipage symptom checker demo with three ML models and local explainability using SHAP and LIME.

## Features
- Landing page with system overview and CTA
- Symptom checker with:
  - Symptom selection (typed aliases + multiselect catalog)
  - Dictionary-based spell correction for typed symptom text
  - Model selection
  - Prediction results with confidence/risk category and low-confidence warning
  - SHAP and LIME explanations
- Disease information, about, and feedback pages

## Project Structure
- `app.py`: landing page
- `pages/`: Streamlit multipage views
- `evaluate_models.py`: CLI model evaluation
- `src/config.py`: project constants and tunables
- `src/data.py`: data loading and symptom parsing
- `src/models.py`: model training and evaluation (Random Forest, Logistic Regression, Naive Bayes)
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
- The checker requires at least two recognized symptoms before prediction to reduce noisy outputs.
