import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, recall_score, precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Directories
PROCESSED_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed"
REPORTS_DIR = r"D:\Internship2026\Infectious-Disease-Triage\reports"
MODELS_DIR = r"D:\Internship2026\Infectious-Disease-Triage\models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    # 1. Load data
    features_file = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found at {features_file}. Please run preprocess.py first.")
        
    df = pd.read_csv(features_file)
    print(f"Loaded feature matrix: {df.shape}")
    
    # 2. Select optimal feature set and target
    features = ['current_sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
    X = df[features]
    y = df['label']
    groups = df['subject_id']
    
    print(f"Features selected for Logistic Regression training: {features}")
    print(f"Target distribution: {y.sum()} positive rows out of {len(y)} ({y.mean()*100:.2f}%)")
    
    # 3. 5-Fold GroupKFold Cross-Validation Setup
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    lr_oof_preds = np.zeros(len(df))
    
    print("Running 5-fold GroupKFold cross-validation for Logistic Regression...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Logistic Regression Pipeline (with Standard Scaling)
        lr_fold = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
        )
        lr_fold.fit(X_train, y_train)
        lr_oof_preds[val_idx] = lr_fold.predict_proba(X_val)[:, 1]
        
    # 4. Evaluate out-of-fold Logistic Regression
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
    
    # 5. Evaluate qSOFA Baseline
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
    
    # Print results
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
    
    # Save metrics report
    report = {
        'logistic_regression': {
            'auroc': lr_auroc,
            'auprc': lr_auprc,
            'sensitivity': lr_sens,
            'specificity': lr_spec,
            'precision': lr_ppv,
            'f1_score': best_lr_f1,
            'threshold': best_lr_threshold,
            'confusion_matrix': {'tp': int(tp_l), 'fp': int(fp_l), 'fn': int(fn_l), 'tn': int(tn_l)}
        },
        'qsofa': {
            'auroc': qsofa_auroc,
            'auprc': qsofa_auprc,
            'sensitivity': qsofa_sens,
            'specificity': qsofa_spec,
            'precision': qsofa_ppv,
            'f1_score': qsofa_f1,
            'confusion_matrix': {'tp': int(tp_q), 'fp': int(fp_q), 'fn': int(fn_q), 'tn': int(tn_q)}
        }
    }
    with open(os.path.join(REPORTS_DIR, "metrics_report_lr.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # 6. Generate Evaluation Plots
    plt.figure(figsize=(18, 5))
    
    # Plot 1: ROC Curves
    plt.subplot(1, 3, 1)
    from sklearn.metrics import roc_curve
    fpr_l, tpr_l, _ = roc_curve(y, lr_oof_preds)
    fpr_q, tpr_q, _ = roc_curve(y, qsofa_pred)
    plt.plot(fpr_l, tpr_l, color='forestgreen', lw=2, label=f'Logistic Reg (OOF AUC = {lr_auroc:.3f})')
    plt.plot(fpr_q, tpr_q, color='navy', lw=2, linestyle='--', label=f'qSOFA Baseline (AUC = {qsofa_auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: PR Curves
    plt.subplot(1, 3, 2)
    plt.plot(lr_recall, lr_precision, color='forestgreen', lw=2, label=f'Logistic Reg (AUPRC = {lr_auprc:.3f})')
    plt.plot(qsofa_rec, qsofa_prec, color='navy', lw=2, linestyle='--', label=f'qSOFA Baseline (AUPRC = {qsofa_auprc:.3f})')
    plt.axhline(y=y.mean(), color='red', linestyle=':', label=f'Random Guess ({y.mean()*100:.2f}%)')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Calibration Curves
    plt.subplot(1, 3, 3)
    prob_true_l, prob_pred_l = calibration_curve(y, lr_oof_preds, n_bins=5, strategy='uniform')
    plt.plot(prob_pred_l, prob_true_l, marker='x', color='forestgreen', label='Logistic Reg')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "evaluation_curves_lr.png"), dpi=150)
    plt.close()
    
    # 7. Train Final Model on All Data
    print("\nTraining final model on all data...")
    final_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
    )
    final_lr.fit(X, y)
    with open(os.path.join(MODELS_DIR, "sepsis_lr.pkl"), "wb") as f:
        pickle.dump(final_lr, f)
    
    print("Final Logistic Regression pipeline saved to models/sepsis_lr.pkl")
    print("Evaluation curves saved to reports/evaluation_curves_lr.png")

if __name__ == "__main__":
    main()
