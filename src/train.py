import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, roc_curve
)

DATA_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\mimic-iv-3.1"
PROCESSED_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed"

features_file = os.path.join(PROCESSED_DIR, "sepsis_features_final.csv")
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file not found at {features_file}. Please run dataclean.py first.")
    
df = pd.read_csv(features_file)
print(f"Loaded feature matrix: {df.shape}")

print("Loading diagnoses to extract target sepsis labels...")
diagnoses = pd.read_csv(os.path.join(DATA_DIR, "hosp", "diagnoses_icd.csv.gz"), usecols=['hadm_id', 'icd_code'])
d_icd = pd.read_csv(os.path.join(DATA_DIR, "hosp", "d_icd_diagnoses.csv.gz"), usecols=['icd_code', 'long_title'])
sepsis_code_list = d_icd[d_icd['long_title'].str.contains('sepsis|septic', case=False, na=False)]['icd_code'].unique()
sepsis_hadms = set(diagnoses[diagnoses['icd_code'].isin(sepsis_code_list)]['hadm_id'])
df['label'] = df['hadm_id'].isin(sepsis_hadms).astype(int)

# Use cleaned vital stats and demographic columns
features = [
    '220045_mean', '220181_mean', '220210_mean', '220277_mean', '223762_mean',
    '220045_max', '220181_max', '220210_max', '220277_max', '223762_max',
    '220045_min', '220181_min', '220210_min', '220277_min', '223762_min',
    '220045_std', '220181_std', '220210_std', '220277_std', '223762_std',
    'anchor_age', 'gender'
]

X = df[features]
y = df['label']
groups = df['subject_id']

print(f"Features selected for XGBoost training: {features}")
print(f"Target distribution: {y.sum()} positive rows out of {len(y)} ({y.mean()*100:.2f}%)")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

xgb_oof_preds = np.zeros(len(df))

print("Running 5-fold GroupKFold cross-validation for XGBoost...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_w = neg_count / pos_count if pos_count > 0 else 1.0
    
    xgb_fold = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.01,
        scale_pos_weight=scale_w,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_fold.fit(X_train, y_train, verbose=False)
    xgb_oof_preds[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
    
xgb_auroc = roc_auc_score(y, xgb_oof_preds)
xgb_precision, xgb_recall, _ = precision_recall_curve(y, xgb_oof_preds)
xgb_auprc = auc(xgb_recall, xgb_precision)

best_xgb_f1 = 0
best_xgb_threshold = 0.5
for th in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(y, (xgb_oof_preds >= th).astype(int))
    if f1 > best_xgb_f1:
        best_xgb_f1 = f1
        best_xgb_threshold = th
        
xgb_preds_binary = (xgb_oof_preds >= best_xgb_threshold).astype(int)
tn_x, fp_x, fn_x, tp_x = confusion_matrix(y, xgb_preds_binary).ravel()
xgb_sens = tp_x / (tp_x + fn_x) if (tp_x + fn_x) > 0 else 0.0
xgb_spec = tn_x / (tn_x + fp_x) if (tn_x + fp_x) > 0 else 0.0
xgb_ppv = tp_x / (tp_x + fp_x) if (tp_x + fp_x) > 0 else 0.0

# Simplified vital-signs-only baseline: Resp Rate >= 22 or MAP <= 70
df['simplified_baseline_score'] = (
    (df['220210_max'] >= 22).astype(int) + 
    (df['220181_min'] <= 70).astype(int)
)
baseline_pred = df['simplified_baseline_score']

baseline_auroc = roc_auc_score(y, baseline_pred)
baseline_prec, baseline_rec, _ = precision_recall_curve(y, baseline_pred)
baseline_auprc = auc(baseline_rec, baseline_prec)

baseline_preds_binary = (df['simplified_baseline_score'] >= 1).astype(int)
tn_q, fp_q, fn_q, tp_q = confusion_matrix(y, baseline_preds_binary).ravel()
baseline_sens = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0.0
baseline_spec = tn_q / (tn_q + fp_q) if (tn_q + fp_q) > 0 else 0.0
baseline_ppv = tp_q / (tp_q + fp_q) if (tp_q + fp_q) > 0 else 0.0
baseline_f1 = f1_score(y, baseline_preds_binary)

print("\n" + "="*20 + " MODEL COMPARISON RESULTS " + "="*20)
print(f"XGBoost (Out-of-Fold, Tuned Threshold = {best_xgb_threshold:.3f}):")
print(f"  AUROC:       {xgb_auroc:.4f}")
print(f"  AUPRC:       {xgb_auprc:.4f}")
print(f"  Sensitivity: {xgb_sens:.4f}")
print(f"  Specificity: {xgb_spec:.4f}")
print(f"  Precision:   {xgb_ppv:.4f}")
print(f"  F1-Score:    {best_xgb_f1:.4f}")
print(f"  Confusion:   TP={tp_x}, FP={fp_x}, FN={fn_x}, TN={tn_x}")
print("-" * 50)
print("Vital-Signs-Only Baseline (Score >= 1):")
print(f"  AUROC:       {baseline_auroc:.4f}")
print(f"  AUPRC:       {baseline_auprc:.4f}")
print(f"  Sensitivity: {baseline_sens:.4f}")
print(f"  Specificity: {baseline_spec:.4f}")
print(f"  Precision:   {baseline_ppv:.4f}")
print(f"  F1-Score:    {baseline_f1:.4f}")
print(f"  Confusion:   TP={tp_q}, FP={fp_q}, FN={fn_q}, TN={tn_q}")
print("="*66)

# Generate visualizations
print("\nGenerating visualizations...")
os.makedirs(r"D:\Internship2026\Infectious-Disease-Triage\reports", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC Curve
xgb_fpr, xgb_tpr, _ = roc_curve(y, xgb_oof_preds)
baseline_fpr, baseline_tpr, _ = roc_curve(y, baseline_pred)
axes[0, 0].plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUROC={xgb_auroc:.4f})', linewidth=2)
axes[0, 0].plot(baseline_fpr, baseline_tpr, label=f'Baseline (AUROC={baseline_auroc:.4f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 0].set_title('ROC Curves', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Precision-Recall Curve
axes[0, 1].plot(xgb_recall, xgb_precision, label=f'XGBoost (AUPRC={xgb_auprc:.4f})', linewidth=2)
axes[0, 1].plot(baseline_rec, baseline_prec, label=f'Baseline (AUPRC={baseline_auprc:.4f})', linewidth=2)
axes[0, 1].set_xlabel('Recall', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Confusion Matrix - XGBoost
cm_xgb = confusion_matrix(y, xgb_preds_binary)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1, 0].set_ylabel('True Label', fontsize=11)
axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
axes[1, 0].set_title('XGBoost Confusion Matrix', fontsize=12, fontweight='bold')

# Confusion Matrix - Baseline
cm_baseline = confusion_matrix(y, baseline_preds_binary)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1], cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1, 1].set_ylabel('True Label', fontsize=11)
axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
axes[1, 1].set_title('Baseline Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(r"D:\Internship2026\Infectious-Disease-Triage\reports", "xgboost_performance.png"), dpi=300, bbox_inches='tight')
print("Saved: xgboost_performance.png")
plt.close()

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
# Train a temporary model to get feature importance
temp_xgb = xgb.XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.01,
    scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1.0,
    reg_alpha=1.0, reg_lambda=5.0, random_state=42, eval_metric='logloss'
)
temp_xgb.fit(X, y)
importance_df = pd.DataFrame({
    'feature': features,
    'importance': temp_xgb.feature_importances_
}).sort_values('importance', ascending=True).tail(15)
ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('XGBoost Top 15 Feature Importance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(r"D:\Internship2026\Infectious-Disease-Triage\reports", "xgboost_feature_importance.png"), dpi=300, bbox_inches='tight')
print("Saved: xgboost_feature_importance.png")
plt.close()

print("\nTraining final model on all data...")
neg_count = len(y) - y.sum()
pos_count = y.sum()
final_scale_w = neg_count / pos_count if pos_count > 0 else 1.0

final_xgb = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.01,
    scale_pos_weight=final_scale_w,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    eval_metric='logloss'
)
final_xgb.fit(X, y)
final_xgb.save_model(os.path.join(PROCESSED_DIR, "sepsis_xgb_model.json"))
print(f"Saved final XGBoost model to {os.path.join(PROCESSED_DIR, 'sepsis_xgb_model.json')}")
print("Training completed successfully!")
