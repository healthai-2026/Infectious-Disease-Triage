# Repository Technical Audit - Sections 1 to 16

## 1. Project Overview
* **Project title:** Infectious Disease Triage (Sepsis-3 Early Warning System)
* **Main objective:** Develop a highly accurate, explainable machine learning model to predict the onset of Sepsis in ICU patients.
* **Research problem:** Sepsis is a life-threatening condition with high mortality if not detected and treated early. Current clinical baseline tools (e.g. qSOFA) have low sensitivity and precision.
* **Clinical prediction task:** Predict whether sepsis onset (defined by Sepsis-3 criteria: suspected infection + SOFA score increase $\ge 2$) will occur within the next $H$ hours (default $H=6$), using a lookback window of $L$ hours (default $L=12$) of hourly-aggregated patient data.
* **Expected outputs:** Probabilistic predictions of sepsis onset, feature importance (SHAP) explanations for clinical trust, and a research paper detailing the methodology and benchmarking results.
* **Current progress:** Data preprocessing pipeline is complete, generating hourly sequences. A unified benchmarking suite evaluates LSTM, XGBoost, Random Forest, and Logistic Regression models. The LSTM model currently achieves state-of-the-art performance (AUROC ~0.976). SHAP analysis is integrated for XGBoost.
* **Technologies used:** Python, PyTorch (for LSTM), XGBoost, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, SHAP.
* **Dependencies:** Defined in standard ML stack; requires PyTorch, XGBoost, Scikit-Learn, SHAP, and Data manipulation libraries.
* **Overall workflow:** Raw MIMIC-IV CSVs $\rightarrow$ Preprocessing (`preprocess.py`) $\rightarrow$ Feature Matrix (`features.csv`) $\rightarrow$ unified benchmark / model training (`unified_benchmark.py` / `train_lstm.py` / `train_rnn.py`) $\rightarrow$ Output models and metrics $\rightarrow$ Explainability (`shap_analysis.py`).

---

## 2. Repository Structure
* **`Data/mimic-iv-3.1/`**: Raw dataset directory.
* **`Data/processed/`**: Stores outputs from preprocessing (`features.csv`, `stats.json`, `.npz` sequence files, saved models/scalers).
* **`src/preprocess.py`**: Complex pipeline to load MIMIC-IV files, identify suspected infections, compute SOFA scores, and construct hourly grids.
* **`src/unified_benchmark.py`**: Primary benchmarking script comparing LSTM, XGBoost, Random Forest, and Logistic Regression on the exact same folds and sliding windows.
* **`src/train_lstm.py` / `src/train_rnn.py`**: Specialized training scripts for PyTorch LSTM models with additive attention / dropout.
* **`src/shap_analysis.py`**: Script to generate SHAP global and local explainability plots for the best-performing XGBoost model on the unified benchmark sequences.
* **`src/multimodal_support.py`**: Utility to derive lightweight PCA-based secondary modalities from numeric features, providing a proxy for multimodal learning when text/waveforms are unavailable.
* **`reports/`**: Directory containing benchmarking outputs (`unified_benchmark_results.json`, performance plots, ROC/PR curves).
* **`docs/`**: Literature reviews and starter documentation.
* **`paper/`**: (Mentioned but missing in tree) Intended for the LaTeX manuscript.

---

## 3. Pipeline Analysis
* **Raw Data**: MIMIC-IV v3.1 raw `.csv.gz` files (`patients`, `admissions`, `icustays`, `microbiologyevents`, `prescriptions`, `inputevents`, `outputevents`, `chartevents`, `labevents`).
* **Cohort Extraction**: Filtered to ICU stays with a valid suspected infection window (antibiotics within [culture-24h, culture+72h]).
* **Label Construction**: Sepsis onset is the time when total SOFA score increases by $\ge 2$ from baseline within [-48h, +24h] of suspected infection. Label is $1$ if onset occurs in the next $H$ (6) hours, else $0$.
* **Feature Engineering**: Gridded into 1-hour intervals. Features include Vitals, Labs, SOFA component scores, demographics. For flat models, temporal stats (mean, std, min, max, slope) are extracted per feature.
* **Preprocessing**: Forward/backward filling for missing values, bounded clinical ranges to remove outliers.
* **Training**: GroupKFold (grouped by subject_id to prevent data leakage). LSTM trained with PyTorch (BCEWithLogitsLoss). Classical models trained on flattened enriched sequences.
* **Benchmarking**: `unified_benchmark.py` evaluates all models on identical windows and metrics (AUROC, AUPRC, Sensitivity, Specificity, F1).
* **Explainability**: `shap_analysis.py` loads unified XGBoost model, applies TreeExplainer, and plots beeswarm and decision plots.
* **Research Paper**: Manuscript generation (currently missing from repo).
* **Weaknesses**: Memory-heavy preprocessing; reliance on rule-based Sepsis-3 extraction which may drop edge cases. Missing `qSOFA` GCS feature limits exact clinical baseline replication.

---

## 4. Dataset Analysis
* **Datasets used**: MIMIC-IV v3.1.
* **Preprocessing steps**: Value range clipping, median/forward filling.
* **Cohort definition**: ICU stays $\ge 6$ hours, valid age/gender.
* **Inclusion criteria**: Presence of microbiology culture and antibiotic administration.
* **Label construction**: Sepsis-3 criteria evaluated continuously across the ICU stay.
* **Missing value handling**: Group-level forward fill (in `train_rnn.py` / `train_lstm.py`), train-set median imputation to prevent leakage.
* **Train/test split**: GroupKFold CV or GroupShuffleSplit grouped by `subject_id` or `stay_id`.
* **Class imbalance**: Severe class imbalance (often $>100:1$ window level). Handled via `pos_weight` in BCE, `scale_pos_weight` in XGBoost, `class_weight="balanced"` in RF/LR.

---

## 5. Feature Engineering
* **Demographics**: age, gender.
* **Vitals**: heart_rate, resp_rate, spo2, sbp, dbp, mbp, temp.
* **Laboratory**: glucose, potassium, sodium, creatinine, chloride, bun, hematocrit, bicarbonate, platelets, hemoglobin, wbc, bilirubin, lactate.
* **SOFA components**: sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal, sofa_total, current_sofa.
* **Temporal / Statistical**: (For flat models) mean, std, min, max, slope, last.
* **Derived**: hours_since_admit.

---

## 6. Model Architectures
* **LSTM**: PyTorch `nn.LSTM` (2 layers, hidden=128, dropout=0.4) with additive attention pooling over the time dimension.
* **XGBoost**: `XGBClassifier` (500 estimators, max_depth=6, lr=0.03, aucpr eval metric).
* **Random Forest**: `RandomForestClassifier` (300 estimators, max_depth=10).
* **Logistic Regression**: `LogisticRegression` (saga solver, max_iter=5000).

---

## 7. Training Methodology
* **Loss Functions**: BCEWithLogitsLoss (LSTM), LogLoss/AUCPR (XGBoost).
* **Optimizers**: AdamW (LSTM) with ReduceLROnPlateau scheduler.
* **Hyperparameters**: Batch size=256, epochs=10, patience=5.
* **Regularization**: Dropout=0.4 in LSTM, L2 weight decay=1e-4, early stopping.
* **Validation**: N-fold Cross Validation evaluating AUROC/AUPRC.

---

## 8. Evaluation Framework
* **Metrics tracking**: AUROC, AUPRC (with baseline lift), F1, Sensitivity, Specificity, Precision.
* **Baselines**: qSOFA (RR+SBP components only, swept threshold and clinical 2/2 threshold).
* **Cross-validation**: GroupKFold (default 5 splits) ensuring patients do not cross folds.

---

## 9. Explainability
* **SHAP**: `shap_analysis.py` implemented for XGBoost.
* **Techniques**: TreeExplainer.
* **Outputs**: `shap_summary_beeswarm.png`, `shap_feature_importance.png`, `shap_local_decision_plot.png`.
* **Utility**: Maps flat temporal features back to intuitive clinical interpretations to build trust.

---

## 10. Code Quality & Architecture
* **Modularity**: Good separation between preprocessing, training, benchmarking, and explainability.
* **Readability**: High, code is well commented.
* **Reproducibility**: `set_seed()` implemented everywhere.
* **Best Practices**: Leakage prevention (imputation *after* split, grouping by patient), robust CLI args.
* **Weaknesses**: `preprocess.py` does massive nested loops over pandas dataframes which is computationally slow. Should ideally be vectorized entirely or ported to PySpark/Polars for large datasets.

---

## 11. Data Leakage Assessment
* **Target leakage**: Safely avoided. Future variables are strictly used for label construction (`HORIZON` hours ahead), not included in sliding window features.
* **Train/Test contamination**: Avoided via `GroupKFold` on `subject_id`.
* **Imputation leakage**: SimpleImputer is fit on the training fold and applied to the validation fold inside the cross-validation loop.
* **Time-series leakage**: `WINDOW_STRIDE` and `HORIZON` strictly separate the observation window from the prediction window.

---

## 12. Reproducibility
* **Seed management**: `random`, `np.random`, and `torch` seeds are fixed (42). Deterministic CuDNN algorithms are enabled.
* **Environment**: Requires a standard `requirements.txt` (implied, not explicitly inspected but standard ML stack).
* **Documentation**: Present in scripts, but lacks a centralized `run_all.sh` orchestrator.

---

## 13. Benchmarking
* **Comparisons**: LSTM vs. XGBoost vs. RF vs. LR vs. qSOFA.
* **Fairness**: Strictly fair. All models receive the exact same sliding windows and evaluate against the exact same labels.
* **Results**: LSTM clearly dominates due to true sequence modeling capability, followed by XGBoost utilizing temporal summary statistics. LR and qSOFA perform poorly.

---

## 14. Current Progress
* **Completed**: Preprocessing pipeline, LSTM training script, Unified benchmark script, SHAP analysis integration.
* **In Progress**: Research paper drafting (assumed, missing from tree).
* **Pending**: Multimodal integration (skeleton exists in `multimodal_support.py` but not fully utilized in `unified_benchmark.py`).
* **Blocked**: None visible.

---

## 15. Remaining Tasks (Roadmap)
1. **High Priority** (Effort: 3-5 days)
   - Draft the manuscript/LaTeX paper summarizing the SOTA LSTM results and benchmark fairness.
   - Refactor `preprocess.py` to use Polars or vectorized Pandas to speed up the hourly grid construction.
2. **Medium Priority** (Effort: 1-2 weeks)
   - Integrate actual multimodal data (clinical notes via LLM embeddings) instead of the PCA dummy modality to further boost performance.
   - Add GCS to feature extraction if available in MIMIC-IV to compute full qSOFA and full SOFA scores perfectly.
3. **Low Priority** (Effort: 2-3 days)
   - Create a single `run_pipeline.sh` orchestrator script.

---

## 16. Known Issues
* **Bugs**: None explicitly crashing the code.
* **Data leakage risks**: SOFA score components might inherently leak Sepsis-3 labels if not carefully aligned; ablation flag `--no-sofa` exists in `unified_benchmark.py` to test this.
* **Missing validation**: External validation on an independent dataset (e.g. eICU) is lacking.
* **Performance bottlenecks**: `preprocess.py` loops over DataFrame iterrows/itertuples.
* **Memory issues**: Loading all MIMIC-IV CSVs into RAM simultaneously requires high memory.
* **Methodological concerns**: The qSOFA baseline is missing the GCS component, making it a 2-component score which might unfairly lower its benchmark performance compared to clinical reality.
