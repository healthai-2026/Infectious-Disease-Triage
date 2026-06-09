import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROCESSED_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed"

features_file = os.path.join(PROCESSED_DIR, "features.csv")
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file not found at {features_file}. Please run preprocess.py first.")
    
df = pd.read_csv(features_file)
print(f"Loaded feature matrix: {df.shape}")

features = ['current_sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
X = df[features]
y = df['label']
groups = df['subject_id']

print(f"Features selected for Logistic Regression training: {features}")
print(f"Target distribution: {y.sum()} positive rows out of {len(y)} ({y.mean()*100:.2f}%)")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

lr_oof_preds = np.zeros(len(df))

print("Running 5-fold GroupKFold cross-validation for Logistic Regression...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    lr_fold = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
    )
    lr_fold.fit(X_train, y_train)
    lr_oof_preds[val_idx] = lr_fold.predict_proba(X_val)[:, 1]
    
lr_auroc = roc_auc_score(y, lr_oof_preds)
lr_precision, lr_recall, _ = precision_recall_curve(y, lr_oof_preds)
lr_auprc = auc(lr_recall, lr_precision)

best_lr_threshold = 0.5
best_lr_f1 = 0
for th in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(y, (lr_oof_preds >= th).astype(int))
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_threshold = th
        
lr_preds_binary = (lr_oof_preds >= best_lr_threshold).astype(int)
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y, lr_preds_binary).ravel()
lr_sens = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
lr_spec = tn_l / (tn_l + fp_l) if (tn_l + fp_l) > 0 else 0.0
lr_ppv = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0

df['qsofa_score'] = (
    (df['resp_rate_last'] >= 22).astype(int) + 
    (df['sbp_last'] <= 100).astype(int) + 
    (df['sofa_cns'] > 0).astype(int)
)
qsofa_pred = df['qsofa_score']

qsofa_auroc = roc_auc_score(y, qsofa_pred)
qsofa_prec, qsofa_rec, _ = precision_recall_curve(y, qsofa_pred)
qsofa_auprc = auc(qsofa_rec, qsofa_prec)

qsofa_preds_binary = (df['qsofa_score'] >= 2).astype(int)
tn_q, fp_q, fn_q, tp_q = confusion_matrix(y, qsofa_preds_binary).ravel()
qsofa_sens = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0.0
qsofa_spec = tn_q / (tn_q + fp_q) if (tn_q + fp_q) > 0 else 0.0
qsofa_ppv = tp_q / (tp_q + fp_q) if (tp_q + fp_q) > 0 else 0.0
qsofa_f1 = f1_score(y, qsofa_preds_binary)

print("\n" + "="*20 + " LOGISTIC REGRESSION RESULTS " + "="*20)
print(f"Logistic Regression (Out-of-Fold, Tuned Threshold = {best_lr_threshold:.3f}):")
print(f"  AUROC:       {lr_auroc:.4f}")
print(f"  AUPRC:       {lr_auprc:.4f}")
print(f"  Sensitivity: {lr_sens:.4f}")
print(f"  Specificity: {lr_spec:.4f}")
print(f"  Precision:   {lr_ppv:.4f}")
print(f"  F1-Score:    {best_lr_f1:.4f}")
print(f"  Confusion:   TP={tp_l}, FP={fp_l}, FN={fn_l}, TN={tn_l}")
print("-" * 50)
print("qSOFA Baseline (qSOFA >= 2):")
print(f"  AUROC:       {qsofa_auroc:.4f}")
print(f"  AUPRC:       {qsofa_auprc:.4f}")
print(f"  Sensitivity: {qsofa_sens:.4f}")
print(f"  Specificity: {qsofa_spec:.4f}")
print(f"  Precision:   {qsofa_ppv:.4f}")
print(f"  F1-Score:    {qsofa_f1:.4f}")
print(f"  Confusion:   TP={tp_q}, FP={fp_q}, FN={fn_q}, TN={tn_q}")
print("="*69)

print("\nTraining final model on all data...")
final_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
)
final_lr.fit(X, y)
print("Training completed successfully!")
