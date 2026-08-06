import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Sepsis-3 Model Results API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR     = os.path.join(ROOT_DIR, "src")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
PROCESSED_DIR = os.path.join(ROOT_DIR, "Data", "processed")

# Serve report images as static files
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# ---------------------------------------------------------------------------
# Script registry — each entry describes a runnable script
# ---------------------------------------------------------------------------
SCRIPTS: dict[str, dict] = {
    "train_models": {
        "label":       "Train XGBoost + Random Forest",
        "description": "Trains XGBoost and Random Forest classifiers on the full feature matrix. Saves metrics to reports/.",
        "file":        os.path.join(SRC_DIR, "train_models.py"),
        "args":        [],
        "color":       "#00c9ff",
        "icon":        "🌳",
        "estimated":   "~10–20 min",
    },
    "train_lr": {
        "label":       "Train Logistic Regression",
        "description": "Trains a Logistic Regression baseline with GroupKFold cross-validation. Saves lr_metrics.json and LR performance plots.",
        "file":        os.path.join(SRC_DIR, "train_lr.py"),
        "args":        [],
        "color":       "#a78bfa",
        "icon":        "📈",
        "estimated":   "~5–10 min",
    },
    "train_rnn": {
        "label":       "Train RNN-LSTM",
        "description": "Trains a bidirectional LSTM on sequential ICU data. Saves rnn_metrics.json and the model checkpoint.",
        "file":        os.path.join(SRC_DIR, "train_rnn.py"),
        "args":        [],
        "color":       "#4ade80",
        "icon":        "🔁",
        "estimated":   "~15–30 min",
    },
    "train": {
        "label":       "Multimodal Ablation (XGBoost)",
        "description": "Full multimodal ablation study: tabular-only, multimodal, modality-only XGBoost variants with GroupKFold.",
        "file":        os.path.join(SRC_DIR, "train.py"),
        "args":        [],
        "color":       "#fb923c",
        "icon":        "🧪",
        "estimated":   "~20–40 min",
    },
    "shap_analysis": {
        "label":       "SHAP Explainability",
        "description": "Runs SHAP TreeExplainer on the trained XGBoost model. Generates summary, beeswarm, waterfall, and dependence plots.",
        "file":        os.path.join(SRC_DIR, "shap_analysis.py"),
        "args":        [],
        "color":       "#f472b6",
        "icon":        "🔍",
        "estimated":   "~2–5 min",
    },
    "train_lstm": {
        "label":       "Train LSTM (sequential)",
        "description": "Trains a standalone LSTM model on sequential patient data. Saves lstm_performance.png and model checkpoint.",
        "file":        os.path.join(SRC_DIR, "train_lstm.py"),
        "args":        [],
        "color":       "#34d399",
        "icon":        "🧠",
        "estimated":   "~15–30 min",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


async def stream_subprocess(script_key: str) -> AsyncIterator[str]:
    """Run a script and yield SSE-formatted lines."""
    info = SCRIPTS[script_key]
    script_path = info["file"]

    if not os.path.exists(script_path):
        yield f"data: [ERROR] Script not found: {script_path}\n\n"
        yield "data: [EXIT:1]\n\n"
        return

    yield f"data: ▶  Starting: {info['label']}\n\n"
    yield f"data: ⏱  Estimated runtime: {info['estimated']}\n\n"
    yield f"data: 📂  Working directory: {ROOT_DIR}\n\n"
    yield "data: " + "─" * 60 + "\n\n"

    cmd = [sys.executable, script_path] + info["args"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ROOT_DIR,
        )

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            # SSE: escape newlines within a single data field
            yield f"data: {line}\n\n"

        await proc.wait()
        yield "data: " + "─" * 60 + "\n\n"
        status = "✅  Completed successfully" if proc.returncode == 0 else f"❌  Exited with code {proc.returncode}"
        yield f"data: {status}\n\n"
        yield f"data: [EXIT:{proc.returncode}]\n\n"

    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [EXIT:1]\n\n"


# ---------------------------------------------------------------------------
# Routes — results
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Sepsis-3 Model Results API v3", "status": "active"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/scripts")
def list_scripts():
    """Return metadata for all runnable scripts."""
    return [
        {
            "key": key,
            "label": v["label"],
            "description": v["description"],
            "color": v["color"],
            "icon": v["icon"],
            "estimated": v["estimated"],
            "available": os.path.exists(v["file"]),
        }
        for key, v in SCRIPTS.items()
    ]


@app.get("/api/run/{script_key}")
async def run_script(script_key: str):
    """Stream script stdout/stderr as Server-Sent Events."""
    if script_key not in SCRIPTS:
        raise HTTPException(status_code=404, detail=f"Unknown script: {script_key}")

    return StreamingResponse(
        stream_subprocess(script_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


@app.get("/api/results/all")
def get_all_results():
    """Return all model metrics, SHAP results, and dataset stats in one call."""
    dataset_stats   = load_json(os.path.join(PROCESSED_DIR, "stats.json")) or {}
    xgb_metrics     = load_json(os.path.join(REPORTS_DIR, "xgboost_metrics.json")) or {}
    lr_metrics      = load_json(os.path.join(REPORTS_DIR, "lr_metrics.json")) or {}
    rnn_metrics     = load_json(os.path.join(REPORTS_DIR, "rnn_metrics.json")) or {}
    metrics_report  = load_json(os.path.join(REPORTS_DIR, "metrics_report.json")) or {}
    metrics_report_lr = load_json(os.path.join(REPORTS_DIR, "metrics_report_lr.json")) or {}

    qsofa = (
        metrics_report.get("qsofa")
        or metrics_report_lr.get("qsofa")
        or {}
    )

    shap_results = load_json(os.path.join(REPORTS_DIR, "shap_results.json")) or {}

    feature_importance = []
    feat_imp_path = os.path.join(REPORTS_DIR, "multimodal_feature_importance.csv")
    if os.path.exists(feat_imp_path):
        with open(feat_imp_path, "r") as f:
            lines = f.read().strip().splitlines()
        for line in lines[1:]:
            vals = line.split(",")
            if len(vals) >= 3:
                feature_importance.append({
                    "feature":          vals[0],
                    "importance_mean":  float(vals[1]),
                    "importance_std":   float(vals[2]),
                })

    images = {}
    image_map = {
        "xgboost_performance":          "xgboost_performance.png",
        "lr_performance":               "lr_performance.png",
        "rnn_performance":              "rnn_performance.png",
        "lstm_performance":             "lstm_performance.png",
        "roc_pr_curves":                "roc_pr_curves.png",
        "roc_pr_comparison":            "roc_pr_comparison.png",
        "confusion_matrices":           "confusion_matrices.png",
        "confusion_matrices_comparison":"confusion_matrices_comparison.png",
        "shap_summary_bar":             "shap_summary_bar.png",
        "shap_summary_beeswarm":        "shap_summary_beeswarm.png",
        "shap_waterfall_positive":      "shap_waterfall_positive.png",
        "shap_dependence_lactate":      "shap_dependence_lactate.png",
        "shap_dependence_sofa":         "shap_dependence_sofa.png",
        "xgboost_feature_importance":   "xgboost_feature_importance.png",
        "feature_importances":          "feature_importances.png",
        "lstm_fold_aurocs":             "lstm_fold_aurocs.png",
        "baseline_performance":         "baseline_performance.png",
        "tabular_only_performance":     "tabular_only_performance.png",
        "multimodal_performance":       "multimodal_performance.png",
        "modality_only_performance":    "modality_only_performance.png",
    }
    for key, filename in image_map.items():
        if os.path.exists(os.path.join(REPORTS_DIR, filename)):
            images[key] = f"/reports/{filename}"

    return {
        "dataset_stats": dataset_stats,
        "models": {
            "xgboost":             xgb_metrics,
            "logistic_regression": lr_metrics,
            "rnn_lstm":            rnn_metrics,
            "qsofa_baseline":      qsofa,
        },
        "shap":               shap_results,
        "feature_importance": feature_importance,
        "images":             images,
        "last_updated":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# Run using: python -m uvicorn backend.main:app --reload
