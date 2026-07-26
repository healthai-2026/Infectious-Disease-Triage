# AI-Enabled Sepsis-3 Onset Prediction

**An AI-Enabled Early Warning System for Sepsis-3 Onset Prediction Using Temporal Deep Learning on the MIMIC-IV Clinical Database**

> Internship Project — *AI-Enabled Clinical Triage and Remote Monitoring for Infectious Disease Care*
> JK Lakshmipat University · May – August 2026

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-MIMIC--IV%20v3.1-orange.svg)](https://physionet.org/content/mimiciv/3.1/)

---

## Overview

Sepsis is a life-threatening syndrome of organ dysfunction caused by a dysregulated host response to infection, responsible for more than 11 million deaths annually worldwide. Current bedside screening tools most notably the quick Sequential Organ Failure Assessment (qSOFA) score — miss more than 62% of at-risk patients before irreversible deterioration occurs.

This repository contains a complete, reproducible machine learning pipeline that predicts **Sepsis-3 onset six hours before clinical recognition**, trained on the full MIMIC-IV clinical database. We compare four model families Logistic Regression, XGBoost, Random Forest, and a temporal LSTM using patient-grouped cross-validation, and show that temporal deep learning substantially outperforms both clinical scoring tools and static machine learning approaches.

**Headline result:** Our LSTM achieves an **AUROC of 0.9761** (AUPRC: 0.6355, Sensitivity: 78.98%, Precision: 57.18%) — a ~19-fold reduction in false-alarm burden compared to the best static model, directly addressing the alert-fatigue problem that limits clinical AI adoption.

---

## Key Results

| Model | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Protocol |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| qSOFA (≥ 2) | 0.481 | 0.007 | 16.3% | 68.6% | 0.5% | 0.009 | Literature |
| Logistic Regression | 0.688 | 0.019 | 13.5% | 96.3% | 3.0% | 0.049 | 5-fold GroupKFold |
| Random Forest | 0.720 | 0.027 | 66.0% | 66.0% | 2.0% | 0.030 | Patient-safe 80/20 split |
| XGBoost | 0.809 | 0.038 | 12.9% | 98.4% | 6.5% | 0.087 | 5-fold GroupKFold |
| **LSTM** | **0.976** | **0.636** | **79.0%** | **97.0%** | **57.2%** | **0.663** | 2-fold GroupKFold |

**Dataset scale:** 94,458 ICU stays · 364,627 patients · 4,571,933 hourly feature rows · 37,994 positive Sepsis-3 labels (0.83% prevalence)

**SHAP explainability** identifies respiratory organ dysfunction (`sofa_resp`) and elevated lactate (`lactate_last`) as the two dominant predictors — consistent with established sepsis pathophysiology.

---

## Repository Structure

```
sepsis-prediction/
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── preprocess.py              # Full MIMIC-IV → hourly feature grid + Sepsis-3 labels
│   ├── train_lr.py                # Logistic Regression (5-fold GroupKFold)
│   ├── train.py                   # XGBoost baseline + multimodal ablation
│   ├── train_models.py            # Random Forest + XGBoost (patient-safe split)
│   ├── train_lstm.py              # Primary LSTM model (2-fold GroupKFold)
│   ├── train_rnn.py               # Secondary RNN pipeline
│   ├── multimodal_support.py      # PCA-based secondary modality module
│   └── shap_analysis.py           # SHAP explainability for XGBoost
│
├── tests/
│   └── test_multimodal_support.py # Regression tests for multimodal pipeline
│
├── Data/
│   ├── mimic-iv-3.1/              # Raw MIMIC-IV tables (not included — see Data Access)
│   └── processed/                 # Generated: features.csv, stats.json, trained models
│
├── reports/                       # Generated: figures, metrics, SHAP outputs
│
├── paper/
│   └── sepsis_paper.tex           # IEEE-format research manuscript
│
└── docs/
    ├── cohort_definition.md       # Clinical cohort & SOFA labeling methodology
    └── literature_review.md       # Annotated bibliography
```

---

## Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/sepsis-prediction.git
cd sepsis-prediction

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Get MIMIC-IV access

This project uses the **MIMIC-IV v3.1** database, which requires credentialed access via PhysioNet:

1. Create an account at [physionet.org](https://physionet.org/)
2. Complete the required CITI training ("Data or Specimens Only Research")
3. Request access to [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
4. Download and place the raw `.csv.gz` files at:
   ```
   Data/mimic-iv-3.1/hosp/
   Data/mimic-iv-3.1/icu/
   ```

> Raw data is **not included** in this repository and must be obtained independently under the PhysioNet Data Use Agreement.

### 3. Run the pipeline

```bash
# Step 1 — Build the hourly feature grid and Sepsis-3 labels
python src/preprocess.py

# Step 2 — Train baseline models
python src/train_lr.py          # Logistic Regression
python src/train_models.py      # Random Forest + XGBoost (patient-safe split)
python src/train.py             # XGBoost (GroupKFold) + multimodal ablation

# Step 3 — Train the primary temporal model
python src/train_lstm.py --plot

# Step 4 — Generate SHAP explainability outputs
python src/shap_analysis.py
```

All metrics, plots, and JSON result summaries are written to `reports/`. Trained model artifacts are saved to `Data/processed/`.

---

## Methodology Summary

- **Cohort:** Adult ICU patients (age ≥ 18), ICU stay ≥ 6 hours, excluding those already septic at admission
- **Outcome:** Sepsis-3 onset SOFA score increase ≥ 2 points from admission baseline **and** suspected infection (culture order + antibiotic administration within a ±24h/72h window)
- **Prediction task:** Binary classification — will the patient meet Sepsis-3 criteria within the next 6 hours?
- **Features:** 62 features per hourly time point 7 SOFA sub-scores, 42 vital-sign statistics (mean/std/min/max/last/slope across 7 vitals), 13 laboratory values, and 2 demographic features
- **Validation:** Patient-grouped cross-validation (`GroupKFold` / `GroupShuffleSplit` on `subject_id`) throughout, to prevent patient-level data leakage

Full methodological detail is available in [`paper/sepsis_paper.tex`](paper/sepsis_paper.tex) and [`docs/cohort_definition.md`](docs/cohort_definition.md).

---

## Known Limitations

- Single-center, retrospective data (MIMIC-IV, one hospital system); external validation on eICU is planned
- Class imbalance (0.83% positive rate) limits AUPRC and threshold stability
- SOFA sub-scores are included as input features, creating partial overlap with the SOFA-based outcome label (disclosed transparently)
- Cross-validation protocol is not yet fully unified across all five models (see paper Limitations for detail)
- Multimodal extension currently uses a PCA-derived proxy signal rather than genuine clinical-note embeddings

See the paper's Limitations and Future Work sections for the complete discussion.

---

## Citation

If you use this code or refer to these results, please cite:

```bibtex
@unpublished{agarwal2026sepsis,
  title  = {An AI-Enabled Early Warning System for Sepsis-3 Onset Prediction
            Using Temporal Deep Learning on the MIMIC-IV Clinical Database},
  author = {Agarwal, Lakshya and Shekhawat, Deepanshu Singh and Sinhal, Amit},
  year   = {2026},
  note   = {JK Lakshmipat University}
}
```

---

## Team

| Name | Role |
|---|---|
| **Lakshya Agarwal** | Research Intern — pipeline, modeling, paper |
| **Deepanshu Singh Shekhawat** | Research Intern — RNN pipeline, multimodal extension |
| **Dr. Amit Sinhal** | Supervisor — Department of Computer Science and Engineering, JK Lakshmipat University |

---

## Acknowledgements

We thank the PhysioNet team and the MIMIC-IV database creators for making de-identified clinical data freely available to the research community, and Dr. Amit Sinhal for guidance and computational resources throughout this project.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details. Note that the MIT License applies to this codebase only; MIMIC-IV data usage is governed separately by the PhysioNet Data Use Agreement.
