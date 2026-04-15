# End-to-End Explainable CKD Risk Prediction & Severity Assessment Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## Abstract

This project implements a complete, end-to-end machine learning pipeline for predicting Chronic Kidney Disease (CKD), assigning probability-based risk scores, providing SHAP-based explainable predictions, and estimating CKD severity through eGFR staging. The system includes an interactive Streamlit web dashboard for clinical decision support.

## Features

- **Binary CKD Classification** using 5 tuned ML models (LightGBM, CatBoost, XGBoost, Random Forest, Stacking Ensemble)
- **Risk Scoring** — Low / Medium / High categories based on prediction probability
- **SHAP Explainability** — Global and per-patient feature importance explanations
- **CKD Severity Staging** — eGFR-based staging (Stages 1–5)
- **Class Imbalance Handling** — SMOTE applied on training data only
- **Interactive Streamlit Dashboard** with 6 pages (Overview, Data Explorer, Model Performance, Risk Assessment, Population Analytics, About)
- **Patient Report Cards** — Comprehensive individual patient summaries
- **Jupyter Notebook** — Full pipeline in notebook format for reproducibility

## Installation

```bash
git clone <repository-url>
cd ckd-risk-prediction
pip install -r requirements.txt
```

## Usage

### Run the full pipeline (CLI)

```bash
python main.py
```

Optional arguments:

```bash
python main.py --data_path data/preprocessed_ckd_data.xlsx --patient_index 5
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

> **Note:** Run `python main.py` first to train models and generate outputs before using the dashboard.

## Project Structure

```
ckd-risk-prediction/
│
├── data/
│   └── preprocessed_ckd_data.xlsx          # Dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Module 1: Data Input
│   ├── preprocessing.py                     # Module 2: Preprocessing
│   ├── model_training.py                    # Module 3: Model Training
│   ├── evaluation.py                        # Module 4: Evaluation
│   ├── risk_scoring.py                      # Module 5: Risk Scoring
│   ├── explainability.py                    # Module 6: SHAP Explainability
│   ├── staging.py                           # Module 7: eGFR & CKD Staging
│   └── reporting.py                         # Module 8: Reporting & Visualization
│
├── models/                                  # Saved trained models (.joblib)
├── outputs/                                 # Generated plots, reports, CSVs
│
├── notebooks/
│   └── CKD_Full_Pipeline.ipynb              # Complete Jupyter notebook
│
├── app.py                                   # Streamlit web dashboard
├── main.py                                  # CLI pipeline runner
├── requirements.txt                         # Dependencies
├── README.md                                # This file
└── .gitignore
```

## Dataset Description

- **Source:** Locally collected clinical data
- **Size:** 500 rows, 18 columns (16 after dropping timestamp and consent columns)
- **Target:** `ckd_diagnosis` (binary: 1 = CKD, 0 = Not CKD)
- **Class Distribution:** 401 CKD (80.2%) vs 99 non-CKD (19.8%) — imbalanced
- **Numerical Features (4):** age, serum_creatinine, egfr, hemoglobin
- **Categorical Features:** diabetes, hypertension, family CKD history, swelling, foamy urine, low urine output, extra salt intake, smoking, painkiller use, living area, gender
- **Preprocessing:** No null values; already cleaned

## Methodology

1. **Data Loading** — Load Excel file, drop irrelevant columns, validate structure
2. **Preprocessing** — Encode categoricals, scale numericals (StandardScaler), stratified 80/20 split, SMOTE on training data
3. **Model Training** — Train 5 models with tuned hyperparameters, save to disk
4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold Cross-Validation; select best model by Recall then F1
5. **Risk Scoring** — Probability thresholds: Low (<0.3), Medium (0.3–0.7), High (≥0.7)
6. **Explainability** — SHAP TreeExplainer for global/local explanations
7. **CKD Staging** — eGFR-based staging per KDIGO guidelines (Stages 1–5)
8. **Reporting** — Patient report cards, correlation heatmaps, feature importance plots

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LightGBM | — | — | — | — | — |
| CatBoost | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Stacking Ensemble | — | — | — | — | — |

> Results will be populated after running the pipeline. See `outputs/model_comparison.csv` for actual values.

## Screenshots

> Screenshots of the Streamlit dashboard can be added here after running the application.

## Future Work

- Integrate additional clinical biomarkers (albumin, BUN, potassium)
- Implement deep learning models (neural networks) for comparison
- Add longitudinal patient tracking for disease progression prediction
- Deploy as a cloud-hosted web application
- Integrate with electronic health record (EHR) systems
- Add multi-language support for the dashboard

## References

- KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of CKD
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- CKD-EPI Creatinine Equation (2021)
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
