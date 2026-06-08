import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, recall_score, precision_score
)
from sklearn.calibration import calibration_curve

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
    
    print(f"Features selected for XGBoost training: {features}")
    print(f"Target distribution: {y.sum()} positive rows out of {len(y)} ({y.mean()*100:.2f}%)")
    
    # 3. 5-Fold GroupKFold Cross-Validation Setup
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    xgb_oof_preds = np.zeros(len(df))
    
    print("Running 5-fold GroupKFold cross-validation for XGBoost...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Calculate scale_pos_weight for this fold's training split
        neg_count = len(y_train) - y_train.sum()
        pos_count = y_train.sum()
        scale_w = neg_count / pos_count if pos_count > 0 else 1.0
        
        xgb_fold = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=1,
            learning_rate=0.01,
            scale_pos_weight=scale_w,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_fold.fit(X_train, y_train, verbose=False)
        xgb_oof_preds[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
        
    # 4. Evaluate out-of-fold XGBoost
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
    
    # Print comparison
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
    print("qSOFA Baseline (qSOFA >= 2):")
    print(f"  AUROC:       {qsofa_auroc:.4f}")
    print(f"  AUPRC:       {qsofa_auprc:.4f}")
    print(f"  Sensitivity: {qsofa_sens:.4f}")
    print(f"  Specificity: {qsofa_spec:.4f}")
    print(f"  Precision:   {qsofa_ppv:.4f}")
    print(f"  F1-Score:    {qsofa_f1:.4f}")
    print(f"  Confusion:   TP={tp_q}, FP={fp_q}, FN={fn_q}, TN={tn_q}")
    print("="*66)
    
    # Save metrics report
    report = {
        'xgboost': {
            'auroc': xgb_auroc,
            'auprc': xgb_auprc,
            'sensitivity': xgb_sens,
            'specificity': xgb_spec,
            'precision': xgb_ppv,
            'f1_score': best_xgb_f1,
            'threshold': best_xgb_threshold,
            'confusion_matrix': {'tp': int(tp_x), 'fp': int(fp_x), 'fn': int(fn_x), 'tn': int(tn_x)}
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
    with open(os.path.join(REPORTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # 6. Generate Evaluation Plots
    plt.figure(figsize=(18, 5))
    
    # Plot 1: ROC Curves
    plt.subplot(1, 3, 1)
    from sklearn.metrics import roc_curve
    fpr_x, tpr_x, _ = roc_curve(y, xgb_oof_preds)
    fpr_q, tpr_q, _ = roc_curve(y, qsofa_pred)
    plt.plot(fpr_x, tpr_x, color='darkorange', lw=2, label=f'XGBoost (OOF AUC = {xgb_auroc:.3f})')
    plt.plot(fpr_q, tpr_q, color='navy', lw=2, linestyle='--', label=f'qSOFA Baseline (AUC = {qsofa_auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: PR Curves
    plt.subplot(1, 3, 2)
    plt.plot(xgb_recall, xgb_precision, color='darkorange', lw=2, label=f'XGBoost (AUPRC = {xgb_auprc:.3f})')
    plt.plot(qsofa_rec, qsofa_prec, color='navy', lw=2, linestyle='--', label=f'qSOFA Baseline (AUPRC = {qsofa_auprc:.3f})')
    plt.axhline(y=y.mean(), color='red', linestyle=':', label=f'Random Guess ({y.mean()*100:.2f}%)')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Calibration Curves
    plt.subplot(1, 3, 3)
    prob_true_x, prob_pred_x = calibration_curve(y, xgb_oof_preds, n_bins=5, strategy='uniform')
    plt.plot(prob_pred_x, prob_true_x, marker='o', color='darkorange', label='XGBoost')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "evaluation_curves.png"), dpi=150)
    plt.close()
    
    # 7. Train Final Model on All Data
    print("\nTraining final model on all data...")
    neg_count = len(y) - y.sum()
    pos_count = y.sum()
    final_scale_w = neg_count / pos_count if pos_count > 0 else 1.0
    
    final_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=1,
        learning_rate=0.01,
        scale_pos_weight=final_scale_w,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        eval_metric='logloss'
    )
    final_xgb.fit(X, y)
    final_xgb.save_model(os.path.join(MODELS_DIR, "sepsis_xgb.json"))
    print("Final XGBoost model saved to models/sepsis_xgb.json")
    
    # 8. Plot and save feature importance (XGBoost)
    importance = final_xgb.feature_importances_
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5))
    feat_imp.plot(kind='barh', color='teal')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('SOFA Score Component')
    plt.title('SOFA Component Importance for Sepsis Prediction (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    
    print("Evaluation curves and feature importance plots saved to reports/")

if __name__ == "__main__":
    main()
