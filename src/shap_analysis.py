"""
SHAP Explainability Analysis for Sepsis-3 XGBoost Model (Unified Benchmark version)
=========================================================
Generates SHAP-based feature importance plots and clinical interpretations.
"""

import os
import json
import warnings
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# Check for required libraries
try:
    import shap
except ImportError:
    print("ERROR: shap library not installed.")
    print("Run: pip install shap")
    exit(1)

try:
    import xgboost as xgb
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    exit(1)

warnings.filterwarnings('ignore')

# Configuration
ROOT_DIR = r"C:\PS1\Infectious-Disease-Triage"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.csv")
MODEL_FILE = os.path.join(PROCESSED_DIR, "unified_xgb_model.json")
IMPUTER_FILE = os.path.join(PROCESSED_DIR, "unified_imputer.pkl")
SCALER_FILE = os.path.join(PROCESSED_DIR, "unified_scaler.pkl")
BENCHMARK_RESULTS = os.path.join(REPORTS_DIR, "unified_benchmark_results.json")

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("SHAP Explainability Analysis for Sepsis-3 Prediction (Unified Benchmark)")
print("=" * 70)

print("\n[1/5] Loading configuration and models...")
if not os.path.exists(BENCHMARK_RESULTS):
    print(f"ERROR: {BENCHMARK_RESULTS} not found.")
    exit(1)
with open(BENCHMARK_RESULTS, "r") as f:
    benchmark_json = json.load(f)
config = benchmark_json["config"]
SEQ_LEN = config["seq_len"]
STRIDE = config["stride"]
HORIZON = config["horizon"]
print(f"  Config: seq_len={SEQ_LEN}, stride={STRIDE}, horizon={HORIZON}")

if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    print("Please run unified_benchmark.py first.")
    exit(1)
if not os.path.exists(IMPUTER_FILE) or not os.path.exists(SCALER_FILE):
    print("ERROR: Imputer or Scaler missing. Please run unified_benchmark.py first.")
    exit(1)

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)
imputer = joblib.load(IMPUTER_FILE)
scaler = joblib.load(SCALER_FILE)
print("  Model, imputer, and scaler loaded.")

print("\n[2/5] Loading data and building windows...")
df = pd.read_csv(FEATURES_FILE)
df = df.dropna(subset=["label"])
df = df.sort_values(["stay_id", "hours_since_admit"]).reset_index(drop=True)

ID_COLS = [c for c in ["stay_id", "subject_id", "hadm_id", "time", "hours_since_admit", "label"] if c in df.columns]
FEAT_COLS = [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]

if len(FEAT_COLS) != config["n_features_raw"]:
    SOFA_COLS = [c for c in FEAT_COLS if "sofa" in c.lower()]
    FEAT_COLS = [c for c in FEAT_COLS if c not in SOFA_COLS]
N_FEAT = len(FEAT_COLS)
print(f"  Using {N_FEAT} raw features.")

Xfull = df[FEAT_COLS].values.astype(np.float32)
yfull = df["label"].values.astype(np.float32)
group_key = "subject_id" if "subject_id" in df.columns else "stay_id"

seq_X, seq_y = [], []
for stay_id, grp in df.groupby("stay_id", sort=False):
    idx = grp.index.tolist()
    n = len(idx)
    if n < SEQ_LEN: continue
    for start in range(0, n - SEQ_LEN + 1, STRIDE):
        end = start + SEQ_LEN
        future_idx = idx[end : end + HORIZON]
        if len(future_idx) == 0: continue
        label = 1.0 if yfull[future_idx].sum() > 0 else 0.0
        seq_X.append(Xfull[idx[start:end]])
        seq_y.append(label)

seq_X = np.array(seq_X, dtype=np.float32)
seq_y = np.array(seq_y, dtype=np.float32)

def add_temporal_stats(X_3d):
    N, T, F = X_3d.shape
    means  = X_3d.mean(axis=1)
    stds   = X_3d.std(axis=1)
    mins   = X_3d.min(axis=1)
    maxs   = X_3d.max(axis=1)
    t      = (np.arange(T, dtype=np.float32) - T / 2.0)
    denom  = float((t ** 2).sum())
    slopes = (X_3d * t[np.newaxis, :, np.newaxis]).sum(axis=1) / denom
    raw    = X_3d.reshape(N, -1)
    return np.concatenate([raw, means, stds, mins, maxs, slopes], axis=1).astype(np.float32)

print("\n[3/5] Applying imputation, scaling, and temporal stats...")
N_WIN = len(seq_X)
seq_X_flat_raw = seq_X.reshape(N_WIN, -1)
X_imp = imputer.transform(seq_X_flat_raw)
X_raw = scaler.transform(X_imp).astype(np.float32)
X_3d = X_raw.reshape(-1, SEQ_LEN, N_FEAT)
X_enr = add_temporal_stats(X_3d)

# Feature Names Generation
enriched_feature_names = []
for t_idx in range(SEQ_LEN):
    for f in FEAT_COLS:
        enriched_feature_names.append(f"{f}_t{t_idx}")
for stat in ["mean", "std", "min", "max", "slope"]:
    for f in FEAT_COLS:
        enriched_feature_names.append(f"{f}_{stat}")

X_enr_df = pd.DataFrame(X_enr, columns=enriched_feature_names)
y_enr_df = pd.Series(seq_y, name="label")

# Stratified Sample
print(f"  Sampling {SAMPLE_SIZE} instances for SHAP...")
actual_sample_size = min(SAMPLE_SIZE, len(X_enr_df))
sss = StratifiedShuffleSplit(n_splits=1, test_size=actual_sample_size, random_state=RANDOM_STATE)
_, sample_idx = next(sss.split(X_enr_df, y_enr_df))
X_sample = X_enr_df.iloc[sample_idx].reset_index(drop=True)
y_sample = y_enr_df.iloc[sample_idx].reset_index(drop=True)

print(f"  Sample shape: {X_sample.shape}, Positive: {y_sample.sum()}")

print("\n[4/5] Computing SHAP values (this may take a moment)...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    model_expected_value = explainer.expected_value
    if isinstance(model_expected_value, list):
        model_expected_value = model_expected_value[1]
except Exception as e:
    print(f"ERROR: Failed to compute SHAP values: {e}")
    exit(1)

print("\n[5/5] Generating plots and reports...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

feature_importance['rank'] = feature_importance.index + 1
feature_importance['cumulative_pct'] = 100 * feature_importance['mean_abs_shap'].cumsum() / feature_importance['mean_abs_shap'].sum()

print("\nTop 10 Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {int(row['rank']):2d}. {row['feature']:30s} | {row['mean_abs_shap']:9.6f}")

try:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    top_feature = feature_importance.iloc[0]['feature']
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, shap_values, X_sample, interaction_index="auto", show=False)
    plt.tight_layout()
    # Clean the feature name for file path in case it has weird characters
    clean_top_feature = "".join([c if c.isalnum() else "_" for c in top_feature])
    plt.savefig(os.path.join(REPORTS_DIR, f"shap_dependence_{clean_top_feature}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    tp_mask = y_sample == 1
    if tp_mask.sum() > 0:
        y_pred_proba = model.predict_proba(X_sample)[:, 1]
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
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_waterfall_positive.png"), dpi=150, bbox_inches='tight')
        plt.close()

except Exception as e:
    print(f"Plotting error: {e}")

text_path = os.path.join(REPORTS_DIR, "shap_feature_importance.txt")
with open(text_path, 'w', encoding='utf-8') as f:
    f.write("SHAP Feature Importance (Unified Benchmark)\n" + "="*50 + "\n")
    for _, row in feature_importance.iterrows():
        f.write(f"{int(row['rank']):4d} | {row['feature']:35s} | {row['mean_abs_shap']:9.6f}\n")

json_path = os.path.join(REPORTS_DIR, "shap_results.json")
top_20 = feature_importance.head(20).to_dict('records')
for item in top_20:
    item['rank'] = int(item['rank'])
    item['mean_abs_shap'] = float(item['mean_abs_shap'])
    item['cumulative_pct'] = float(item['cumulative_pct'])
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({'top_20_features': top_20, 'sample_size': actual_sample_size}, f, indent=2)

print("\nDone! SHAP outputs saved to reports/")
