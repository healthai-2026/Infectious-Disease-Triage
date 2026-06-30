# Sepsis Model Performance Outputs

Below is a summary of the performance metrics for the models trained in this repository, extracted from the generated reports.

## Random Forest (Latest Output)
- **AUROC**: 0.8252
- **AUPRC**: 0.5231
- **Sensitivity (Recall)**: 0.72
- **Precision**: 0.40
- **F1 Score**: 0.51
- **Confusion Matrix**: 
  - True Positives (TP): 2482
  - False Positives (FP): 3728
  - False Negatives (FN): 947
  - True Negatives (TN): 11731

## XGBoost (Latest Output)
- **AUROC**: 0.8453
- **AUPRC**: 0.5629
- **Sensitivity (Recall)**: 0.76
- **Precision**: 0.41
- **F1 Score**: 0.53
- **Confusion Matrix**: 
  - True Positives (TP): 2620
  - False Positives (FP): 3756
  - False Negatives (FN): 809
  - True Negatives (TN): 11703

## Logistic Regression (Prior Output)
- **AUROC**: 0.6734
- **AUPRC**: 0.0128
- **Sensitivity**: 0.7551
- **Specificity**: 0.6053
- **Precision**: 0.0173
- **F1 Score**: 0.0339
- **Confusion Matrix**: 
  - True Positives (TP): 37
  - False Positives (FP): 2098
  - False Negatives (FN): 12
  - True Negatives (TN): 3217

## qSOFA (Clinical Baseline)
- **AUROC**: 0.4807
- **AUPRC**: 0.0072
- **Sensitivity**: 0.1633
- **Specificity**: 0.6858
- **Precision**: 0.0048
- **F1 Score**: 0.0093
- **Confusion Matrix**: 
  - True Positives (TP): 8
  - False Positives (FP): 1670
  - False Negatives (FN): 41
  - True Negatives (TN): 3645

> **Note on RNN/LSTM:** The `src/train_rnn.py` script requires a pre-processed sequential dataset file named `sepsis_sequential.npz`, which currently does not exist in the repository (the `extract_data.py` script only generates the flat `sepsis_features.csv`). Therefore, `train_rnn.py` cannot be run successfully at this moment until that `.npz` file is supplied or generated.
