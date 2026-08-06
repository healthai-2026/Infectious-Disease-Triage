# AI-Enabled Sepsis-3 Onset Prediction

**An AI-Enabled Early Warning System for Sepsis-3 Onset Prediction Using Temporal Deep Learning on the MIMIC-IV Clinical Database**

> Practice School-I Internship Project — *AI-Enabled Clinical Triage and Remote Monitoring for Infectious Disease Care*
> JK Lakshmipat University · Department of Computer Science and Engineering · May – August 2026
> Supervisor: Dr. Amit Sinhal

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-MIMIC--IV%20v3.1-orange.svg)](https://physionet.org/content/mimiciv/3.1/)

---

## Overview

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Current bedside screening tools — most notably the quick Sequential Organ Failure Assessment (qSOFA) score — are severely limited in ICU populations, operating at near-random AUROC and missing the vast majority of at-risk patients before irreversible deterioration.

This repository implements a complete, reproducible machine learning pipeline that predicts **Sepsis-3 onset six hours in advance**, trained and evaluated on the full MIMIC-IV v3.1 clinical database. Four model families are compared — Logistic Regression, Random Forest, XGBoost, and a temporal attention-LSTM — under a **strictly unified benchmark** that enforces identical preprocessing, identical cross-validation folds, identical sliding windows, and identical evaluation metrics for all models, eliminating the experimental confounders that plague multi-model comparison studies in the clinical ML literature.

---

## Key Results (5-Fold GroupKFold, Definitive Run)

| Model | AUROC | AUPRC | AUPRC Lift | Sensitivity | Specificity | F1 | Protocol |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| qSOFA (≥ 2 criteria) | 0.5012 | 0.0060 | 1.0x | 6.0% | 93.5% | 0.010 | Clinical rule (RR + SBP) |
| Logistic Regression | 0.7554 | 0.0209 | 3.5x | 11.5% | 98.4% | 0.059 | 5-fold GroupKFold |
| LSTM | 0.7664 | 0.0245 | 4.1x | 10.8% | 98.7% | 0.067 | 5-fold GroupKFold |
| Random Forest | 0.7786 | 0.0231 | 3.9x | 10.0% | 98.7% | 0.061 | 5-fold GroupKFold |
| **XGBoost** | **0.7935** | **0.0275** | **4.6x** | **13.2%** | **98.4%** | **0.070** | 5-fold GroupKFold |

**Dataset scale:** 4,571,933 hourly feature rows · 1,279,595 sliding windows · 7,648 positive labels (0.60% window prevalence) · 64 raw features per time step · 1,088 enriched features (flat models)

**XGBoost best fold:** Fold 3 AUROC = 0.8095. All models evaluated on identical out-of-fold predictions from the same 5-fold GroupKFold split by `subject_id`.

> Sensitivity/Specificity/F1 are reported at the F1-optimal data-driven threshold. AUPRC baseline (random classifier) = 0.0060 (= window prevalence).

---

## Repository Structure

```
Infectious-Disease-Triage/
│
├── src/
│   ├── preprocess.py               # MIMIC-IV → hourly feature grid + Sepsis-3 labels
│   ├── unified_benchmark.py        # Central pipeline: all models, identical folds/windows
│   ├── shap_analysis.py            # SHAP explainability (consumes unified_benchmark JSON)
│   ├── generate_report_figures.py  # Publication figures from benchmark output
│   ├── train_lstm.py               # Standalone LSTM (5-vital, 2-fold, lighter config)
│   ├── train_rnn.py                # Secondary RNN pipeline with RF/XGB comparison
│   ├── train_lr.py                 # Standalone Logistic Regression
│   ├── train_models.py             # Standalone RF + XGBoost
│   ├── train.py                    # XGBoost baseline + multimodal ablation
│   └── multimodal_support.py       # PCA-based secondary modality module
│
├── Data/
│   ├── mimic-iv-3.1/               # Raw MIMIC-IV tables (not included — see Data Access)
│   │   ├── hosp/                   # patients, admissions, labevents, prescriptions, …
│   │   └── icu/                    # icustays, chartevents, inputevents, outputevents
│   └── processed/                  # Generated: features.csv, unified_xgb_model.json, …
│
├── reports/                        # Generated figures and JSON metrics
│   ├── fig_roc_comparison.png
│   ├── fig_pr_comparison.png
│   ├── fig_fold_aurocs.png
│   ├── fig_mean_auroc_errorbars.png
│   ├── fig_auroc_auprc_bar.png
│   ├── fig_confusion_matrices.png
│   ├── fig_lstm_training_curve.png
│   ├── fig_operating_points.png
│   └── unified_benchmark_results.json
│
├── docs/
│   └── LitratureReview1.md         # Literature review notes (Futoma et al. MGP-RNN)
│
├── PS1_Internship_Report.md        # Full 25-page PS-I internship report
├── PROJECT_AUDIT.md                # Deep technical audit of the repository
├── MASTER_PROJECT_CONTEXT.md       # AI-agent context document for future sessions
├── requirements.txt
└── LICENSE
```

---

## Pipeline Overview

```
Raw MIMIC-IV v3.1 (9 tables)
        │
        ▼
preprocess.py
  ├── Cohort extraction (ICU stays ≥ 6 h with suspected infection)
  ├── Hourly feature grid (forward-fill → backward-fill → median imputation)
  ├── Physiological range clipping (11 vital-sign filters)
  ├── SOFA computation (6 organ systems from charted values)
  ├── Sepsis-3 label: SOFA ↑≥2 within ±48/24h of suspected infection
  └── features.csv  (4,571,933 rows × 65 columns)
        │
        ▼
unified_benchmark.py                          ← PRIMARY PIPELINE
  ├── Sliding windows (SEQ_LEN=12h, STRIDE=3h, HORIZON=6h)
  ├── Temporal statistics enrichment (mean, std, min, max, slope per feature)
  ├── 5-fold GroupKFold by subject_id
  ├── Within-fold imputation (median) + scaling (StandardScaler) — no leakage
  ├── Models: LSTM · XGBoost · Random Forest · Logistic Regression · qSOFA
  ├── OOF evaluation: AUROC · AUPRC · F1-optimal · fixed sensitivity operating points
  └── Saves: unified_benchmark_results.json · unified_xgb_model.json · imputer · scaler
        │
        ▼
shap_analysis.py
  └── TreeExplainer on best-fold XGBoost → beeswarm, dependence, waterfall plots

generate_report_figures.py
  └── 8 publication-quality figures from benchmark JSON output
```

---

## Methodology

### Cohort Definition
Adult ICU patients from MIMIC-IV v3.1 with a stay of at least 6 hours and a valid suspected infection event. Suspected infection requires a microbiology culture order and an antibiotic administration within a [−24 h, +72 h] window around the culture time.

### Sepsis-3 Label
Sepsis onset is identified as the first hour within [−48 h, +24 h] of the suspected infection time at which the SOFA total increases by ≥ 2 points from the window baseline. A sliding window is labelled **positive** if the onset time falls within the 6-hour horizon following the window end.

### Features (64 per time step)
- **42 vital-sign statistics** — 7 vitals (HR, RR, SpO2, SBP, DBP, MAP, Temp) × 6 stats (mean, std, min, max, last, slope) over a 6-hour lookback
- **13 laboratory values** — glucose, potassium, sodium, creatinine, chloride, BUN, haematocrit, bicarbonate, platelets, haemoglobin, WBC, bilirubin, lactate (last value in lookback)
- **7 SOFA scores** — total + 6 organ system sub-scores at the current hour
- **2 demographics** — age, sex

### Unified Benchmark Design
The key methodological contribution is strict experimental uniformity across all models:
- All models receive **identical sliding windows** (12-h lookback, 3-h stride, 6-h horizon)
- All models use **identical 5-fold GroupKFold splits** (grouped by `subject_id`)
- Classical models receive an **enriched flat representation** (raw 768 + temporal stats 320 = 1,088 features) so they can access temporal information without requiring sequence processing
- **Imputation and scaling are fit on training folds only** and applied to validation folds, preventing data leakage
- All models are evaluated using **identical out-of-fold predictions** and identical threshold selection

### Model Configurations

| Model | Key Hyperparameters |
|---|---|
| **LSTM** | 2-layer, hidden=128, attention pooling, dropout=0.4, AdamW lr=3e-4, BCEWithLogitsLoss(pos_weight), ReduceLROnPlateau, early-stop patience=5 |
| **XGBoost** | 500 trees, depth=6, lr=0.03, subsample=0.8, colsample=0.8, eval=aucpr, early-stop=20 rounds |
| **Random Forest** | 300 trees, max_depth=10, min_samples_split=5, class_weight=balanced |
| **Logistic Regression** | saga solver, max_iter=5000, class_weight=balanced |
| **qSOFA** | RR ≥ 22 + SBP ≤ 100 (2-component; GCS excluded from feature matrix) |

---

## Getting Started

### 1. Clone and set up

```bash
git clone https://github.com/healthai-2026/Infectious-Disease-Triage.git
cd Infectious-Disease-Triage

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Obtain MIMIC-IV access

This project requires **MIMIC-IV v3.1**, which requires credentialed access via PhysioNet:

1. Create an account at [physionet.org](https://physionet.org/)
2. Complete the CITI "Data or Specimens Only Research" training
3. Request access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
4. Download and place the raw `.csv.gz` files at:
   ```
   Data/mimic-iv-3.1/hosp/
   Data/mimic-iv-3.1/icu/
   ```

> Raw data is **not included** in this repository and must be obtained independently under the PhysioNet Data Use Agreement.

### 3. Run the pipeline

```bash
# Step 1 — Build the hourly feature grid and Sepsis-3 labels (~hours on first run)
python src/preprocess.py

# Step 2 — Run the unified benchmark (primary pipeline)
python src/unified_benchmark.py

# Step 3 — Generate publication figures from benchmark output
python src/generate_report_figures.py

# Step 4 — SHAP explainability (requires Step 2 artifacts)
python src/shap_analysis.py
```

All metrics and JSON results are written to `reports/`. Trained model artifacts are saved to `Data/processed/`.

---

## Outputs

### Benchmark Results (`reports/unified_benchmark_results.json`)
Contains configuration, per-fold AUROCs, and all aggregate OOF metrics for every model.

### Figures (`reports/`)

| File | Description |
|---|---|
| `fig_roc_comparison.png` | ROC curves for all models through actual operating points |
| `fig_pr_comparison.png` | Precision-Recall curves with prevalence baseline |
| `fig_fold_aurocs.png` | Per-fold AUROC — 5 folds × 5 models |
| `fig_mean_auroc_errorbars.png` | Mean OOF AUROC ± 1 std |
| `fig_auroc_auprc_bar.png` | AUROC and AUPRC side-by-side comparison |
| `fig_confusion_matrices.png` | Confusion matrices at F1-optimal threshold |
| `fig_lstm_training_curve.png` | LSTM training dynamics for all 5 folds |
| `fig_operating_points.png` | Specificity at 50/70/80% sensitivity targets |

### Model Artifacts (`Data/processed/`)

| File | Description |
|---|---|
| `features.csv` | Full hourly feature matrix with Sepsis-3 labels |
| `unified_xgb_model.json` | Best-fold XGBoost model (Fold 3, AUROC=0.8095) |
| `unified_imputer.pkl` | Fitted median imputer from best fold |
| `unified_scaler.pkl` | Fitted StandardScaler from best fold |

---

## Known Limitations

- **Single-centre data** — MIMIC-IV covers one hospital system (BIDMC); external validation on eICU is planned
- **SOFA-label overlap** — SOFA sub-scores are included as input features, creating partial structural overlap with the SOFA-based label (disclosed; ablation available via `--no-sofa` flag)
- **qSOFA incompleteness** — GCS is not retained in the feature matrix; only the 2-component RR + SBP qSOFA is evaluated
- **No probability calibration** — Calibration curves and Expected Calibration Error are not yet reported
- **Multimodal extension** — `multimodal_support.py` uses PCA-derived proxies rather than genuine clinical note embeddings
- **No demographic fairness analysis** — Subgroup performance by age, sex, and ethnicity has not been evaluated

---

## Team

| Name | Role |
|---|---|
| **Lakshya Agarwal** | Research Intern — preprocessing, unified benchmark, report |
| **Deepanshu Singh Shekhawat** | Research Intern — RNN pipeline, multimodal extension |
| **Dr. Amit Sinhal** | Supervisor — Department of Computer Science and Engineering, JKLU |

---

## Acknowledgements

We thank the PhysioNet team and the MIMIC-IV database creators for making de-identified clinical data freely available to the research community, and Dr. Amit Sinhal for guidance and computational resources throughout this project.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details. The MIT License applies to this codebase only; MIMIC-IV data usage is governed separately by the PhysioNet Data Use Agreement.
