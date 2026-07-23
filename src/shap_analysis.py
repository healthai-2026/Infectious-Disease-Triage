"""
SHAP Explainability Analysis for Sepsis-3 XGBoost Model
=========================================================
Generates SHAP-based feature importance plots and clinical interpretations.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check for required libraries
try:
    import shap
except ImportError:
    print("ERROR: shap library not installed.")
    print("Run: pip install shap")
    exit(1)

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    exit(1)

warnings.filterwarnings('ignore')

# Configuration
ROOT_DIR = r"C:\PS1\Infectious-Disease-Triage"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.csv")
MODEL_FILE = os.path.join(PROCESSED_DIR, "sepsis_xgb_model.json")

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

# Clinical feature name mapping
CLINICAL_MAPPING = {
    'lactate_last': "Lactate level (tissue hypoperfusion marker)",
    'current_sofa': "Total SOFA score (organ dysfunction severity)",
    'sofa_resp': "Respiratory SOFA (lung dysfunction)",
    'sofa_renal': "Renal SOFA (kidney dysfunction)",
    'sofa_coag': "Coagulation SOFA (clotting dysfunction)",
    'sofa_cns': "CNS SOFA (neurological dysfunction / GCS)",
    'sofa_liver': "Hepatic SOFA (liver dysfunction / bilirubin)",
    'sofa_cardio': "Cardiovascular SOFA (cardiovascular dysfunction / MAP / vasopressors)",
    'heart_rate_slope': "Heart rate trend (rising = deterioration)",
    'resp_rate_slope': "Respiratory rate trend",
    'sbp_slope': "Blood pressure trend (falling = shock risk)",
    'spo2_min': "Minimum SpO2 (worst oxygenation in window)",
    'wbc_last': "White blood cell count (infection marker)",
}

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("SHAP Explainability Analysis for Sepsis-3 Prediction")
print("=" * 70)

# ============================================================================
# [1/5] LOAD DATA AND MODEL
# ============================================================================
print("\n[1/5] Loading data and model...")

if not os.path.exists(FEATURES_FILE):
    print(f"ERROR: Features file not found at {FEATURES_FILE}")
    print("Please run preprocess.py first.")
    exit(1)

if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    print("Expected: {MODEL_FILE}")
    exit(1)

# Load features
print(f"  Loading features from {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop identifier columns
exclude_cols = {'stay_id', 'subject_id', 'time', 'hadm_id', 'hours_since_admit'}
drop_cols = [c for c in exclude_cols if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')

# Separate features and labels
y = df['label'].astype(int)
X = df.drop(columns=['label'])

print(f"  Features: {X.shape[1]} columns")
print(f"  Target distribution: {y.sum():,} positive ({y.mean()*100:.2f}%)")

# Load model
print(f"  Loading XGBoost model from {MODEL_FILE}")
model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)
print(f"  Model loaded successfully")

# ============================================================================
# [2/5] STRATIFIED SAMPLE FOR SHAP COMPUTATION
# ============================================================================
print("\n[2/5] Sampling data for SHAP computation...")

sss = StratifiedShuffleSplit(n_splits=1, test_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
_, sample_idx = next(sss.split(X, y))

X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y.iloc[sample_idx].reset_index(drop=True)

n_positive = y_sample.sum()
print(f"  Sample size: {len(X_sample):,} rows")
print(f"  Positive samples: {n_positive:,} ({y_sample.mean()*100:.2f}%)")
print(f"  Feature columns: {list(X_sample.columns[:5])}... ({X_sample.shape[1]} total)")

# ============================================================================
# [3/5] COMPUTE SHAP VALUES
# ============================================================================
print("\n[3/5] Computing SHAP values (this may take 1-2 minutes)...")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values may be a list; use the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    model_expected_value = explainer.expected_value
    if isinstance(model_expected_value, list):
        model_expected_value = model_expected_value[1]
    
    print(f"  SHAP values computed: shape {shap_values.shape}")
    print(f"  Model expected value: {model_expected_value:.4f}")
except Exception as e:
    print(f"ERROR: Failed to compute SHAP values: {e}")
    exit(1)

# ============================================================================
# [4/5] FEATURE IMPORTANCE RANKING
# ============================================================================
print("\n[4/5] Computing feature importance...")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

feature_importance['rank'] = feature_importance.index + 1
feature_importance['cumulative_pct'] = (
    100 * feature_importance['mean_abs_shap'].cumsum() / feature_importance['mean_abs_shap'].sum()
)

print("\nTop 10 Features by Mean Absolute SHAP Value:")
print("=" * 80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {int(row['rank']):2d}. {row['feature']:30s} | "
          f"Mean |SHAP| = {row['mean_abs_shap']:.6f} | Cumulative = {row['cumulative_pct']:6.2f}%")
print("=" * 80)

# ============================================================================
# [5/5] GENERATE PLOTS
# ============================================================================
print("\n[5/5] Generating plots...")

plot_count = 0
plot_errors = []

# Plot 1: SHAP Summary Beeswarm
try:
    print("  Generating Plot 1/5: SHAP Summary Beeswarm...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=20, show=False)
    plt.title("SHAP Feature Importance — Top 20 Predictors of Sepsis-3 Onset", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_count += 1
    print(f"    [OK] Saved: {plot_path}")
except Exception as e:
    msg = f"Plot 1 (beeswarm) failed: {e}"
    print(f"    [ERROR] {msg}")
    plot_errors.append(msg)

# Plot 2: SHAP Summary Bar
try:
    print("  Generating Plot 2/5: SHAP Summary Bar...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.title("Mean Absolute SHAP Values — Feature Importance Ranking", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "shap_summary_bar.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_count += 1
    print(f"    [OK] Saved: {plot_path}")
except Exception as e:
    msg = f"Plot 2 (bar) failed: {e}"
    print(f"    [ERROR] {msg}")
    plot_errors.append(msg)

# Plot 3: Dependence Plot - Lactate (or top feature)
try:
    print("  Generating Plot 3/5: SHAP Dependence Plot (Lactate)...")
    dep_feature = 'lactate_last' if 'lactate_last' in X_sample.columns else feature_importance.iloc[0]['feature']
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(dep_feature, shap_values, X_sample, interaction_index="auto", show=False)
    plt.title(f"SHAP Dependence Plot — {dep_feature}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "shap_dependence_lactate.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_count += 1
    print(f"    [OK] Saved: {plot_path}")
except Exception as e:
    msg = f"Plot 3 (dependence lactate) failed: {e}"
    print(f"    [ERROR] {msg}")
    plot_errors.append(msg)

# Plot 4: Dependence Plot - SOFA (or second top feature)
try:
    print("  Generating Plot 4/5: SHAP Dependence Plot (SOFA)...")
    dep_feature2 = 'current_sofa' if 'current_sofa' in X_sample.columns else feature_importance.iloc[1]['feature']
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(dep_feature2, shap_values, X_sample, interaction_index="auto", show=False)
    plt.title(f"SHAP Dependence Plot — {dep_feature2}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "shap_dependence_sofa.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_count += 1
    print(f"    [OK] Saved: {plot_path}")
except Exception as e:
    msg = f"Plot 4 (dependence sofa) failed: {e}"
    print(f"    [ERROR] {msg}")
    plot_errors.append(msg)

# Plot 5: Waterfall Plot - Highest Confidence True Positive
try:
    print("  Generating Plot 5/5: SHAP Waterfall (True Positive)...")
    
    # Find highest confidence true positive
    y_pred_proba = model.predict_proba(X_sample)[:, 1]
    tp_mask = y_sample == 1
    
    if tp_mask.sum() > 0:
        tp_indices = np.where(tp_mask)[0]
        best_tp_idx = tp_indices[np.argmax(y_pred_proba[tp_indices])]
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[best_tp_idx],
                           base_values=model_expected_value,
                           data=X_sample.iloc[best_tp_idx].values,
                           feature_names=X_sample.columns.tolist()),
            max_display=15,
            show=False
        )
        plt.title("SHAP Waterfall — Highest Confidence Sepsis Prediction (True Positive)", 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "shap_waterfall_positive.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print(f"    [OK] Saved: {plot_path}")
    else:
        print(f"    [WARNING] No true positive samples in this batch; skipping waterfall plot")
        
except Exception as e:
    msg = f"Plot 5 (waterfall) failed: {e}"
    print(f"    [ERROR] {msg}")
    plot_errors.append(msg)

print(f"\n  Plots generated: {plot_count}/5")
if plot_errors:
    print(f"  Errors encountered:")
    for err in plot_errors:
        print(f"    - {err}")

# ============================================================================
# SAVE TEXT OUTPUT
# ============================================================================
print("\nSaving feature importance report...")

text_path = os.path.join(REPORTS_DIR, "shap_feature_importance.txt")
with open(text_path, 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write("SHAP Feature Importance Report — Sepsis-3 Prediction Model\n")
    f.write("=" * 90 + "\n\n")
    
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Sample Size: {len(X_sample):,}\n")
    f.write(f"Positive Samples: {n_positive:,} ({y_sample.mean()*100:.2f}%)\n")
    f.write(f"Model Expected Value: {model_expected_value:.6f}\n\n")
    
    f.write("Rank | Feature Name                       | Mean |SHAP| | Cumulative %\n")
    f.write("-" * 90 + "\n")
    
    for _, row in feature_importance.iterrows():
        rank = int(row['rank'])
        feature = row['feature']
        mean_shap = row['mean_abs_shap']
        cum_pct = row['cumulative_pct']
        f.write(f"{rank:4d} | {feature:34s} | {mean_shap:9.6f}  | {cum_pct:7.2f}%\n")

print(f"[OK] Saved: {text_path}")

# ============================================================================
# SAVE JSON OUTPUT
# ============================================================================
print("Saving SHAP results JSON...")

json_path = os.path.join(REPORTS_DIR, "shap_results.json")
top_20 = feature_importance.head(20).to_dict('records')
top_20_clean = [
    {
        'rank': int(item['rank']),
        'feature': item['feature'],
        'mean_abs_shap': float(item['mean_abs_shap']),
        'cumulative_pct': float(item['cumulative_pct'])
    }
    for item in top_20
]

results = {
    'top_20_features': top_20_clean,
    'sample_size': int(len(X_sample)),
    'positive_samples': int(n_positive),
    'model_expected_value': float(model_expected_value),
    'computation_date': datetime.now().strftime('%Y-%m-%d'),
    'total_features': int(X_sample.shape[1])
}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"[OK] Saved: {json_path}")

# ============================================================================
# CLINICAL INTERPRETATION
# ============================================================================
print("\n" + "=" * 90)
print("CLINICAL INTERPRETATION")
print("=" * 90)
print("\nTop 5 Predictors of Sepsis-3 Onset (6-hour horizon):\n")

for idx, row in feature_importance.head(5).iterrows():
    rank = int(row['rank'])
    feature = row['feature']
    mean_shap = row['mean_abs_shap']
    clinical = CLINICAL_MAPPING.get(feature, feature)
    print(f"  {rank}. {feature}")
    print(f"     Mean |SHAP|: {mean_shap:.6f}")
    print(f"     Clinical: {clinical}\n")

print("=" * 90)
print(f"\nDone! All outputs saved to {REPORTS_DIR}/")
print("=" * 90)
