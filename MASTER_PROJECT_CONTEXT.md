# Master Project Context (Section 19)

## Complete Project Overview
The "Infectious Disease Triage" project is an advanced Machine Learning research repository aimed at creating a Sepsis-3 Early Warning System (EWS). Using the MIMIC-IV v3.1 dataset, the pipeline processes raw clinical data into hourly time-series grids. The primary objective is to predict sepsis onset (defined clinically via Sepsis-3 criteria: infection + SOFA rise) 6 hours in advance using a 12-hour lookback window.

## Repository Architecture
* **`Data/`**: Contains raw MIMIC-IV tables and `processed/` outputs (`features.csv`, `.npz` files, saved models).
* **`src/`**: Core logic scripts:
  * `preprocess.py`: Extracts cohort, computes labels, constructs hourly grids.
  * `train_lstm.py` / `train_rnn.py`: PyTorch deep learning models.
  * `unified_benchmark.py`: Strict, fair benchmarking across all architectures.
  * `shap_analysis.py`: Global/local interpretability using SHAP.
  * `multimodal_support.py`: PCA-based feature embedding for multimodal proxying.
* **`reports/`**: JSON results, saved scalers, and PNG plots.
* **`docs/`**: Literature reviews and starter context.

## Datasets, Preprocessing, & Feature Engineering
* **Dataset**: MIMIC-IV v3.1.
* **Preprocessing**: Strict rules applied for range-clipping vitals and labs. Group-level forward and backward filling used to handle sparsity.
* **Features**: Hourly vitals (HR, RR, SpO2, BP, Temp), Labs, Demographics (Age, Gender), and SOFA sub-scores. For classical models, temporal aggregates (mean, min, max, std, slope) are computed per window to capture trajectory.

## Models, Evaluation, & Benchmarks
* **Models**: PyTorch LSTM (Top performer: AUROC ~0.976), XGBoost, Random Forest, Logistic Regression.
* **Evaluation**: GroupKFold cross-validation (preventing patient leakage). Evaluated on sliding windows predicting $H=6$ hours ahead.
* **Metrics**: AUROC, AUPRC (with baseline lift), F1, Sensitivity, Specificity, Precision.
* **Explainability**: SHAP TreeExplainer integrated for XGBoost, producing beeswarm and decision plots to build clinical trust.

## Paper Status & Known Issues
* **Paper Status**: The manuscript (`paper/`) directory is currently missing from the repository. Drafting the LaTeX paper is a high-priority pending task.
* **Known Issues**: Preprocessing is computationally slow due to pandas iterators; qSOFA baseline is missing the GCS component; external validation is currently absent.

## Roadmap & Future Plans
1. **Manuscript**: Draft and finalize the research paper summarizing the SOTA performance.
2. **Optimization**: Port `preprocess.py` to Polars/PySpark for speed.
3. **Multimodal**: Extend `multimodal_support.py` to ingest real clinical BERT embeddings of nursing notes.
4. **External Validation**: Test the trained models on eICU or an external hospital dataset.

## Implementation Decisions to Remember
* **Leakage Prevention**: Imputation is strictly fitted on the training fold and applied to the validation fold. Splitting is done by `subject_id`, not `stay_id`.
* **Clinical Realism**: Predictions are made on sliding windows rather than static stay-aggregates to simulate real-time hospital deployment.
* **Fairness**: `unified_benchmark.py` forces LSTM and classical models to use the exact same windows and targets, ensuring metric differences are purely algorithmic.
