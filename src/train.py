import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, train_test_split

from multimodal_support import infer_id_col, prepare_multimodal_frame

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "mimic-iv-3.1")
PROCESSED_DIR = os.path.join(ROOT_DIR, "Data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sepsis models with optional multimodal ablation and explainability")
    parser.add_argument("--features-file", default=os.path.join(PROCESSED_DIR, "features.csv"))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--simulate-federated", action="store_true")
    parser.add_argument("--modality-components", type=int, default=4)
    return parser.parse_args()


def select_feature_columns(df: pd.DataFrame, id_col: str, include_modality: bool = False) -> List[str]:
    exclude_cols = {"label", id_col, "hours_since_admit", "site"}
    features = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.startswith("modality_embed_") and not include_modality:
            continue
        if col.startswith("modality_embed_") or col in {"age", "gender"} or pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def build_label(df: pd.DataFrame, id_col: str) -> pd.Series:
    if "label" in df.columns:
        print("Using existing label column from features.csv")
        label = df["label"].dropna().astype(int)
        if len(label) != len(df):
            label = pd.Series(label.reindex(df.index, fill_value=0).values, index=df.index)
        return label

    print("Loading diagnoses to extract target sepsis labels...")
    diagnoses = pd.read_csv(os.path.join(DATA_DIR, "hosp", "diagnoses_icd.csv.gz"), usecols=["hadm_id", "icd_code"])
    d_icd = pd.read_csv(os.path.join(DATA_DIR, "hosp", "d_icd_diagnoses.csv.gz"), usecols=["icd_code", "long_title"])
    sepsis_code_list = d_icd[d_icd["long_title"].str.contains("sepsis|septic", case=False, na=False)]["icd_code"].unique()
    sepsis_hadms = set(diagnoses[diagnoses["icd_code"].isin(sepsis_code_list)]["hadm_id"])
    return df[id_col].isin(sepsis_hadms).astype(int)


def make_model(scale_w: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.02,
        scale_pos_weight=scale_w,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        eval_metric="logloss",
    )


def evaluate_oof(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, float]]]:
    splitter = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        neg_count = int((len(y_train) - y_train.sum()))
        pos_count = int(y_train.sum())
        scale_w = neg_count / pos_count if pos_count > 0 else 1.0
        model = make_model(scale_w)
        model.fit(X_train, y_train, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    metrics = summarize_metrics(y, oof_preds)
    return oof_preds, metrics, None


def summarize_metrics(y: pd.Series, probs: np.ndarray) -> Dict[str, float]:
    if np.unique(y).size <= 1:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "threshold": 0.5,
        }

    precision, recall, _ = precision_recall_curve(y, probs)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y, probs)
    _ = (fpr, tpr)
    best_threshold = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y, (probs >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    preds = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(auprc),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "f1": float(best_f1),
        "threshold": float(best_threshold),
    }


def build_baseline(df: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    resp_col = next((c for c in ["resp_rate_max", "resp_rate_mean", "resp_rate_min", "220210_max"] if c in df.columns), None)
    map_col = next((c for c in ["mbp_min", "mbp_mean", "220181_min"] if c in df.columns), None)
    baseline_score = pd.Series(0, index=df.index, dtype=int)
    if resp_col is not None:
        baseline_score = baseline_score + (df[resp_col] >= 22).astype(int)
    if map_col is not None:
        baseline_score = baseline_score + (df[map_col] <= 70).astype(int)
    baseline_pred = baseline_score.to_numpy(dtype=float)
    return baseline_pred, summarize_metrics(y, baseline_pred)


def explain_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    perm = permutation_importance(model, X, y, n_repeats=8, random_state=42, scoring="roc_auc")
    importance_df = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    return importance_df.head(15)


def simulate_federated(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Dict[str, Dict[str, float]]:
    site_labels = ["mimic_iv" if int(hash(str(val)) % 2) == 0 else "eicu" for val in df[groups.name].astype(str)]
    site_a_idx = np.array([i for i, site in enumerate(site_labels) if site == "mimic_iv"])
    site_b_idx = np.array([i for i, site in enumerate(site_labels) if site == "eicu"])

    preds = np.zeros(len(df))
    metrics = {}
    for site_name, site_idx in [("mimic_iv", site_a_idx), ("eicu", site_b_idx)]:
        if len(site_idx) < 10:
            continue
        site_train_idx, site_test_idx = train_test_split(site_idx, test_size=0.3, random_state=42, stratify=y.iloc[site_idx])
        model = make_model(1.0)
        model.fit(X.iloc[site_train_idx], y.iloc[site_train_idx], verbose=False)
        site_probs = model.predict_proba(X.iloc[site_test_idx])[:, 1]
        metrics[site_name] = summarize_metrics(y.iloc[site_test_idx], site_probs)
        preds[site_test_idx] = site_probs

    if not metrics:
        return {"mimic_iv": {"auroc": float("nan")}, "eicu": {"auroc": float("nan")}}

    agg_metrics = summarize_metrics(y, preds)
    metrics["aggregated"] = agg_metrics
    return metrics


def save_results(results: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def plot_metrics(results: Dict[str, Dict[str, float]], y: pd.Series, probs: np.ndarray, name: str) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    if np.unique(y).size > 1:
        fpr, tpr, _ = roc_curve(y, probs)
        axes[0, 0].plot(fpr, tpr, label=f"{name} (AUROC={results['auroc']:.3f})", linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y, probs)
    axes[0, 1].plot(recall, precision, label=f"{name} (AUPRC={results['auprc']:.3f})", linewidth=2)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].grid(alpha=0.3)

    preds = (probs >= results["threshold"]).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False, xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")
    axes[1, 0].set_title(f"{name} Confusion Matrix")

    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name.lower().replace(' ', '_')}_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.features_file):
        raise FileNotFoundError(f"Features file not found at {args.features_file}. Please run preprocess.py first.")

    print(f"Loading processed features from {args.features_file}")
    df = pd.read_csv(args.features_file)
    print(f"Loaded feature matrix: {df.shape}")

    id_col = infer_id_col(df)
    if id_col is None:
        raise KeyError("No usable patient/stay identifier column found in features.csv")
    print(f"Using identifier column: {id_col}")

    df = df.copy()
    df["label"] = build_label(df, id_col)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    multimodal_df = prepare_multimodal_frame(df, output_dir=PROCESSED_DIR)
    multimodal_df["label"] = df["label"].astype(int)
    multimodal_df["site"] = "mimic_iv"

    base_features = select_feature_columns(multimodal_df, id_col=id_col, include_modality=False)
    modality_features = [c for c in multimodal_df.columns if c.startswith("modality_embed_")]
    multimodal_features = base_features + modality_features

    if len(base_features) < 2:
        raise ValueError(f"No compatible training features found in {args.features_file}.")

    X_base = multimodal_df[base_features]
    X_multimodal = multimodal_df[multimodal_features]
    X_modality_only = multimodal_df[modality_features]
    y = multimodal_df["label"]
    groups = multimodal_df[id_col]

    variants = {
        "tabular_only": X_base,
        "multimodal": X_multimodal,
        "modality_only": X_modality_only,
    }

    results = {}
    for variant_name, X_variant in variants.items():
        oof_preds, metrics, _ = evaluate_oof(X_variant, y, groups, n_splits=args.n_splits)
        results[variant_name] = metrics
        results[f"{variant_name}_oof"] = oof_preds.tolist()
        plot_metrics(metrics, y, oof_preds, variant_name.replace("_", " ").title())

    baseline_pred, baseline_metrics = build_baseline(multimodal_df, y)
    results["baseline"] = baseline_metrics
    plot_metrics(baseline_metrics, y, baseline_pred, "Baseline")

    if args.simulate_federated:
        fed_results = simulate_federated(multimodal_df, X_multimodal, y, groups)
        results["federated"] = fed_results

    ablation = {
        "auroc_delta_multimodal_vs_tabular": results["multimodal"].get("auroc", float("nan")) - results["tabular_only"].get("auroc", float("nan")),
        "auprc_delta_multimodal_vs_tabular": results["multimodal"].get("auprc", float("nan")) - results["tabular_only"].get("auprc", float("nan")),
        "auroc_delta_modality_only_vs_tabular": results["modality_only"].get("auroc", float("nan")) - results["tabular_only"].get("auroc", float("nan")),
    }
    results["ablation"] = ablation

    final_model = make_model(scale_w=max(1.0, int((len(y) - y.sum())) / max(int(y.sum()), 1)))
    final_model.fit(X_multimodal, y, verbose=False)
    importance_df = explain_model(final_model, X_multimodal, y)
    importance_df.to_csv(os.path.join(REPORTS_DIR, "multimodal_feature_importance.csv"), index=False)
    save_results(results, os.path.join(REPORTS_DIR, "multimodal_ablation_results.json"))

    print("\nMultimodal ablation summary")
    for variant_name in ["tabular_only", "multimodal", "modality_only", "baseline"]:
        metrics = results[variant_name]
        print(f"- {variant_name}: AUROC={metrics.get('auroc', float('nan')):.3f}, AUPRC={metrics.get('auprc', float('nan')):.3f}, F1={metrics.get('f1', float('nan')):.3f}")
    print("Ablation deltas:", ablation)
    print("Saved explainability summary to reports/multimodal_feature_importance.csv")
    print("Saved evaluation summary to reports/multimodal_ablation_results.json")


if __name__ == "__main__":
    main()
