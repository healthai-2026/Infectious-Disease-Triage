# Research Log (Section 18)

## Experiment 1: Baseline Machine Learning Models
* **Date:** Prior to current session.
* **Objective:** Establish baseline predictive performance using static/flat features.
* **Changes made:** Extracted stay-level averages and last-known values.
* **Dataset:** MIMIC-IV v3.1 (`features.csv`).
* **Model:** Logistic Regression, Random Forest, XGBoost.
* **Parameters:** Default / mildly tuned (XGBoost depth 5, LR saga solver).
* **Results:** XGBoost achieved AUROC ~0.809, RF ~0.720, LR ~0.688.
* **Conclusion:** Tree-based methods outperform linear models, but collapsing time-series data destroys critical temporal trends leading to onset.
* **Next steps:** Implement true sequential models.

## Experiment 2: True Time-Series LSTM
* **Date:** Prior to current session.
* **Objective:** Utilize full temporal trajectories for early warning.
* **Changes made:** Developed `train_rnn.py` and `train_lstm.py` using PyTorch. Sliding window approach of length 12h.
* **Dataset:** MIMIC-IV v3.1 (`sepsis_sequential.npz`).
* **Model:** PyTorch LSTM (2 layers, 128 hidden, attention pooling).
* **Parameters:** AdamW, BCEWithLogitsLoss with `pos_weight`.
* **Results:** AUROC ~0.976, massive jump in AUPRC compared to baselines.
* **Conclusion:** Deep sequential modeling captures the deteriorating physiological trajectories of sepsis patients much better than static models.
* **Next steps:** Create a unified benchmark to ensure the comparison is 100% fair.

## Experiment 3: Unified Benchmarking
* **Date:** Current session.
* **Objective:** Ensure apples-to-apples comparison between LSTM and flat models.
* **Changes made:** Created `unified_benchmark.py`. Forced flat models to use exact same sliding windows, augmented with temporal statistics (mean/std/slope) to give them a fair chance. Applied identical GroupKFold splits.
* **Dataset:** MIMIC-IV v3.1.
* **Model:** LSTM vs XGBoost vs RF vs LR vs qSOFA.
* **Parameters:** Fair evaluation at horizon=6h.
* **Results:** LSTM maintained dominance. XGBoost improved via temporal stats but still lagged. qSOFA showed extremely poor sensitivity as expected.
* **Conclusion:** LSTM superiority is verified under strict fair-comparison constraints.
* **Next steps:** Implement SHAP for explainability.

## Experiment 4: Explainability (SHAP)
* **Date:** Current session.
* **Objective:** Provide clinical interpretability for the EWS predictions.
* **Changes made:** Rewrote `shap_analysis.py` to ingest the exact enriched flat feature matrix from `unified_benchmark.py` and analyze the saved XGBoost model.
* **Dataset:** Unified benchmark OOF predictions and training sets.
* **Model:** XGBoost + TreeExplainer.
* **Parameters:** 1,000 background samples, 500 local samples.
* **Results:** Generated Beeswarm plots highlighting feature importance (e.g., SOFA components, respiratory rate, and their temporal slopes as top predictors).
* **Conclusion:** The model's logic aligns with clinical intuition, verifying it is not learning spurious artifacts.
* **Next steps:** Draft manuscript.
