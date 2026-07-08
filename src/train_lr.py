import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

PROCESSED_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\processed"

features_file = os.path.join(PROCESSED_DIR, "features.csv")
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file not found at {features_file}. Please run preprocess.py first.")
    
df = pd.read_csv(features_file)
print(f"Loaded feature matrix: {df.shape}")

# Use the pre-computed label from features.csv.
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

id_col = 'subject_id' if 'subject_id' in df.columns else 'stay_id' if 'stay_id' in df.columns else None
if id_col is None:
    raise KeyError("No usable patient/stay identifier column found in features.csv")

exclude_cols = {'label', id_col, 'hadm_id', 'hours_since_admit', 'stay_id', 'time'}
features = [
    col for col in df.columns
    if col not in exclude_cols and (col in {'age', 'gender'} or pd.api.types.is_numeric_dtype(df[col]))
]

X = df[features]
y = df['label']
groups = df[id_col]

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
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
    )
    lr_fold.fit(X_train, y_train)
    lr_oof_preds[val_idx] = lr_fold.predict_proba(X_val)[:, 1]
    
if np.unique(y).size > 1:
    lr_auroc = roc_auc_score(y, lr_oof_preds)
    lr_precision, lr_recall, _ = precision_recall_curve(y, lr_oof_preds)
    lr_auprc = auc(lr_recall, lr_precision)
else:
    lr_auroc = float('nan')
    lr_precision = np.array([0.0])
    lr_recall = np.array([0.0])
    lr_auprc = float('nan')

best_lr_threshold = 0.5
best_lr_f1 = 0
for th in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(y, (lr_oof_preds >= th).astype(int), zero_division=0)
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_threshold = th
        
lr_preds_binary = (lr_oof_preds >= best_lr_threshold).astype(int)
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y, lr_preds_binary).ravel()
lr_sens = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
lr_spec = tn_l / (tn_l + fp_l) if (tn_l + fp_l) > 0 else 0.0
lr_ppv = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0

print("\n" + "="*20 + " LOGISTIC REGRESSION RESULTS " + "="*20)
print(f"Logistic Regression (Out-of-Fold, Tuned Threshold = {best_lr_threshold:.3f}):")
print(f"  AUROC:       {lr_auroc:.4f}")
print(f"  AUPRC:       {lr_auprc:.4f}")
print(f"  Sensitivity: {lr_sens:.4f}")
print(f"  Specificity: {lr_spec:.4f}")
print(f"  Precision:   {lr_ppv:.4f}")
print(f"  F1-Score:    {best_lr_f1:.4f}")
print(f"  Confusion:   TP={tp_l}, FP={fp_l}, FN={fn_l}, TN={tn_l}")
print("="*69)

# Generate visualizations
print("\nGenerating visualizations...")
REPORTS_DIR = r"C:\PS1\Infectious-Disease-Triage\reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Logistic Regression — Sepsis Prediction Performance', fontsize=14, fontweight='bold', y=1.02)

# ROC Curve
if np.unique(y).size > 1:
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_oof_preds)
else:
    lr_fpr, lr_tpr = np.array([]), np.array([])

if len(lr_fpr) > 0:
    axes[0, 0].plot(lr_fpr, lr_tpr, color='darkorange', label=f'LR (AUROC={lr_auroc:.4f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 0].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Precision-Recall Curve
axes[0, 1].plot(lr_recall, lr_precision, color='steelblue', label=f'LR (AUPRC={lr_auprc:.4f})', linewidth=2)
axes[0, 1].axhline(y=y.mean(), color='gray', linestyle='--', label=f'Baseline prevalence ({y.mean():.3f})', linewidth=1)
axes[0, 1].set_xlabel('Recall', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Confusion Matrix
cm_lr = confusion_matrix(y, lr_preds_binary)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0], cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1, 0].set_ylabel('True Label', fontsize=11)
axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Feature Importance via LR Coefficients (train a temp pipeline for this)
temp_lr = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
)
temp_lr.fit(X, y)
coef = temp_lr.named_steps['logisticregression'].coef_[0]
coef_df = pd.DataFrame({'feature': features, 'coefficient': coef})
coef_df = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=True).index)
colors = ['firebrick' if c < 0 else 'steelblue' for c in coef_df['coefficient']]
axes[1, 1].barh(coef_df['feature'], coef_df['coefficient'], color=colors)
axes[1, 1].axvline(x=0, color='black', linewidth=0.8)
axes[1, 1].set_xlabel('Coefficient Value', fontsize=11)
axes[1, 1].set_title('LR Feature Coefficients\n(blue=positive, red=negative)', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "lr_performance.png"), dpi=300, bbox_inches='tight')
print("Saved: lr_performance.png")
plt.close()

print("\nTraining final model on all data...")
final_lr = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
)
final_lr.fit(X, y)

import joblib
joblib.dump(final_lr, os.path.join(PROCESSED_DIR, "sepsis_lr_model.pkl"))
print(f"Saved final LR model to {os.path.join(PROCESSED_DIR, 'sepsis_lr_model.pkl')}")
print("Training completed successfully!")
