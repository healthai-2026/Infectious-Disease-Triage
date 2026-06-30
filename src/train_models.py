import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for running headlessly
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

# Configurations
DATA_PATH = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed\sepsis_features.csv"
OUTPUT_DIR = r"D:\Internship2026\Infectious-Disease-Triage\reports"

def clean_race(race):
    if not isinstance(race, str):
        return 'UNKNOWN/OTHER'
    race_upper = race.upper()
    if 'WHITE' in race_upper or 'PORTUGUESE' in race_upper:
        return 'WHITE'
    elif 'BLACK' in race_upper:
        return 'BLACK'
    elif 'HISPANIC' in race_upper or 'LATINO' in race_upper or 'SOUTH AMERICAN' in race_upper:
        return 'HISPANIC'
    elif 'ASIAN' in race_upper:
        return 'ASIAN'
    else:
        return 'UNKNOWN/OTHER'

def main():
    print("Step 1: Loading extracted sepsis features...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Target variable
    y = df['sepsis']
    
    # Feature variables (drop identifier columns)
    X = df.drop(columns=['stay_id', 'subject_id', 'hadm_id', 'sepsis'])
    
    print("\nStep 2: Preprocessing categorical variables...")
    # Simplify race
    X['race'] = X['race'].apply(clean_race)
    print("Race distribution after cleaning:")
    print(X['race'].value_counts())
    
    # One-hot encoding
    categorical_cols = ['gender', 'race', 'admission_type']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Explicitly convert bool columns from get_dummies to int (0/1) for models like XGBoost
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Separate count and value columns
    # We should fill missing count columns with 0, since no measurement implies count = 0
    count_cols = [c for c in X.columns if c.endswith('_count')]
    other_cols = [c for c in X.columns if not c.endswith('_count')]
    
    X[count_cols] = X[count_cols].fillna(0)
    
    print(f"Total features after encoding and processing counts: {X.shape[1]}")
    
    print("\nStep 3: Splitting into stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} (Sepsis rate: {y_train.mean()*100:.2f}%)")
    print(f"Test set size: {X_test.shape[0]} (Sepsis rate: {y_test.mean()*100:.2f}%)")
    
    print("\nStep 4: Imputing missing values in clinical features...")
    # Median imputer fitted only on Train set and applied to both Train and Test
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    X_train_imputed[other_cols] = imputer.fit_transform(X_train[other_cols])
    X_test_imputed[other_cols] = imputer.transform(X_test[other_cols])
    
    # Verify no missing values remain
    assert X_train_imputed.isnull().sum().sum() == 0, "Missing values remain in train set!"
    assert X_test_imputed.isnull().sum().sum() == 0, "Missing values remain in test set!"
    
    print("\nStep 5: Training classifiers...")
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 1. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    print("Training Random Forest model...")
    rf.fit(X_train_imputed, y_train)
    
    # 2. XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    print("Training XGBoost model...")
    xgb.fit(X_train_imputed, y_train)
    
    print("\nStep 6: Evaluating models on Test set...")
    # Predictions
    rf_pred = rf.predict(X_test_imputed)
    rf_prob = rf.predict_proba(X_test_imputed)[:, 1]
    
    xgb_pred = xgb.predict(X_test_imputed)
    xgb_prob = xgb.predict_proba(X_test_imputed)[:, 1]
    
    print("\n=== Random Forest Performance ===")
    print(classification_report(y_test, rf_pred))
    
    print("=== XGBoost Performance ===")
    print(classification_report(y_test, xgb_pred))
    
    # Metrics calculation
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
    rf_pr_auc = average_precision_score(y_test, rf_prob)
    
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_prob)
    xgb_pr_auc = average_precision_score(y_test, xgb_prob)
    
    print(f"Random Forest ROC-AUC: {rf_auc:.4f} | PR-AUC: {rf_pr_auc:.4f}")
    print(f"XGBoost ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_pr_auc:.4f}")
    
    # Generate Plots
    print("\nStep 7: Generating evaluation plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    axes[0].plot(rf_fpr, rf_tpr, color='#1f77b4', lw=2.5, label=f'Random Forest (AUC = {rf_auc:.3f})')
    axes[0].plot(xgb_fpr, xgb_tpr, color='#ff7f0e', lw=2.5, label=f'XGBoost (AUC = {xgb_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", frameon=True)
    
    # PR Curve
    axes[1].plot(rf_recall, rf_precision, color='#1f77b4', lw=2.5, label=f'Random Forest (AP = {rf_pr_auc:.3f})')
    axes[1].plot(xgb_recall, xgb_precision, color='#ff7f0e', lw=2.5, label=f'XGBoost (AP = {xgb_pr_auc:.3f})')
    axes[1].axhline(y=y_test.mean(), color='gray', lw=1.5, linestyle='--', label=f'Baseline (AP = {y_test.mean():.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Precision (PPV)', fontsize=12)
    axes[1].set_title('Precision-Recall (PR) Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plot_path_curves = os.path.join(OUTPUT_DIR, "roc_pr_curves.png")
    plt.savefig(plot_path_curves, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC & PR curves to {plot_path_curves}")
    
    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rf_cm = confusion_matrix(y_test, rf_pred)
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xticklabels(['No Sepsis', 'Sepsis'])
    axes[0].set_yticklabels(['No Sepsis', 'Sepsis'])
    axes[0].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
    
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xticklabels(['No Sepsis', 'Sepsis'])
    axes[1].set_yticklabels(['No Sepsis', 'Sepsis'])
    axes[1].set_title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path_cm = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(plot_path_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrices to {plot_path_cm}")
    
    # Plot 3: Feature Importance (top 15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # RF
    rf_importances = pd.Series(rf.feature_importances_, index=X_train_imputed.columns)
    rf_top15 = rf_importances.sort_values(ascending=False).head(15)
    sns.barplot(x=rf_top15.values, y=rf_top15.index, ax=axes[0], color='#1f77b4')
    axes[0].set_title('Random Forest Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance Score', fontsize=12)
    
    # XGB
    xgb_importances = pd.Series(xgb.feature_importances_, index=X_train_imputed.columns)
    xgb_top15 = xgb_importances.sort_values(ascending=False).head(15)
    sns.barplot(x=xgb_top15.values, y=xgb_top15.index, ax=axes[1], color='#ff7f0e')
    axes[1].set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Importance Score', fontsize=12)
    
    plt.tight_layout()
    plot_path_fi = os.path.join(OUTPUT_DIR, "feature_importances.png")
    plt.savefig(plot_path_fi, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Feature Importances to {plot_path_fi}")
    
    # Write summary text file
    summary_path = os.path.join(OUTPUT_DIR, "model_results.txt")
    with open(summary_path, 'w') as f:
        f.write("=== SEPSIS MODEL PERFORMANCE EVALUATION ===\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Stays Analyzed: {len(df)}\n")
        f.write(f"Train Set Size: {len(X_train)} (Sepsis rate: {y_train.mean()*100:.2f}%)\n")
        f.write(f"Test Set Size: {len(X_test)} (Sepsis rate: {y_test.mean()*100:.2f}%)\n\n")
        
        f.write("--- RANDOM FOREST ---\n")
        f.write(f"ROC-AUC: {rf_auc:.5f}\n")
        f.write(f"PR-AUC (Average Precision): {rf_pr_auc:.5f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {rf_cm[0,0]}  FP: {rf_cm[0,1]}\n")
        f.write(f"  FN: {rf_cm[1,0]}  TP: {rf_cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, rf_pred))
        f.write("\n")
        
        f.write("--- XGBOOST ---\n")
        f.write(f"ROC-AUC: {xgb_auc:.5f}\n")
        f.write(f"PR-AUC (Average Precision): {xgb_pr_auc:.5f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {xgb_cm[0,0]}  FP: {xgb_cm[0,1]}\n")
        f.write(f"  FN: {xgb_cm[1,0]}  TP: {xgb_cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, xgb_pred))
        
    print(f"\nStep 8: Model results written to {summary_path}")

if __name__ == "__main__":
    main()
