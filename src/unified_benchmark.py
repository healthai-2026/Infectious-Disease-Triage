"""
unified_benchmark.py
--------------------
Fair apples-to-apples comparison: LSTM vs. XGBoost vs. RF vs. LR.

All four models share:
  1. Same evaluation unit   - one prediction per SLIDING WINDOW (seq_len hours)
  2. Same feature set       - all 64 features + temporal stats from features.csv
  3. Same fold structure    - GroupKFold split on subject_id
  4. Same label definition  - will sepsis onset occur in next HORIZON hours?
  5. Same imbalance handling - weighted loss / scale_pos_weight / class_weight

LSTM   : consumes the full (seq_len x 64) tensor with attention pooling
XGBoost: flattened raw sequence + per-feature temporal stats (mean/std/min/max/slope)
RF     : same enriched flat vector
LR     : same enriched flat vector

Run (full):
    python src/unified_benchmark.py --horizon 6 --epochs 10 --plot
Quick smoke test:
    python src/unified_benchmark.py --max-windows 200000 --epochs 3
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import random
import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, roc_curve,
    precision_recall_curve
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------- #
#  CLI                                                                          #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--seq-len",     type=int,  default=12,   help="Lookback window length (hours)")
parser.add_argument("--stride",      type=int,  default=3,    help="Stride between windows")
parser.add_argument("--horizon",     type=int,  default=6,    help="Prediction horizon: sepsis in next N hours")
parser.add_argument("--epochs",      type=int,  default=10,   help="LSTM training epochs per fold")
parser.add_argument("--n-folds",     type=int,  default=5,    help="GroupKFold folds")
parser.add_argument("--batch",       type=int,  default=256,  help="LSTM batch size")
parser.add_argument("--max-windows", type=int,  default=None, help="Cap total windows (quick test, e.g. 200000)")
parser.add_argument("--no-sofa",     action="store_true",     help="Exclude SOFA columns (leakage ablation)")
parser.add_argument("--plot",        action="store_true",     help="Save comparison plots")
parser.add_argument("--save-oof",    action="store_true",     help="Save OOF prediction arrays to NPZ for bootstrap CI")
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#  Paths & Config                                                               #
# ---------------------------------------------------------------------------- #
PROCESSED_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\processed"
REPORTS_DIR   = r"C:\PS1\Infectious-Disease-Triage\reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SEQ_LEN  = args.seq_len
STRIDE   = args.stride
HORIZON  = args.horizon
N_FOLDS  = args.n_folds
EPOCHS   = args.epochs
BATCH    = args.batch
LR_RATE  = 3e-4    # lower LR stabilises training on 167:1 imbalance
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.4    # stronger regularisation
PATIENCE = 5
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device       : " + str(DEVICE))
print("Seq-len      : " + str(SEQ_LEN) + "h  |  Stride: " + str(STRIDE) +
      "h  |  Horizon: " + str(HORIZON) + "h  |  Folds: " + str(N_FOLDS) +
      "  |  LSTM epochs: " + str(EPOCHS))

# ---------------------------------------------------------------------------- #
#  1. LOAD features.csv                                                         #
# ---------------------------------------------------------------------------- #
print("\n[1] Loading features.csv ...")
df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
df = df.dropna(subset=["label"])
df = df.sort_values(["stay_id", "hours_since_admit"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  |  Stay-level positive rate: {df['label'].mean()*100:.2f}%")

ID_COLS = [c for c in ["stay_id", "subject_id", "hadm_id", "time",
                        "hours_since_admit", "label"] if c in df.columns]
FEAT_COLS = [c for c in df.columns
             if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]

# SOFA ablation: optionally exclude all 7 SOFA-derived columns
SOFA_COLS = [c for c in FEAT_COLS if "sofa" in c.lower()]
if args.no_sofa:
    FEAT_COLS = [c for c in FEAT_COLS if c not in SOFA_COLS]
    print(f"    SOFA ablation: removed {len(SOFA_COLS)} cols -> {len(FEAT_COLS)} remaining")
else:
    print(f"    SOFA cols included: {SOFA_COLS}")

N_FEAT = len(FEAT_COLS)
print(f"    Feature columns: {N_FEAT}")

# ---------------------------------------------------------------------------- #
#  2. IMPUTE globally (median)                                                  #
# ---------------------------------------------------------------------------- #
print("\n[2] Skipping global imputation (moved to fold-level to prevent leakage)...")
Xfull = df[FEAT_COLS].values.astype(np.float32)  # (N_rows, N_FEAT)
yfull = df["label"].values.astype(np.float32)

group_key = "subject_id" if "subject_id" in df.columns else "stay_id"

# qSOFA indices: use LAST-step values (clinician's current state at prediction time)
# Using _last suffix features which encode the value at the most recent charted hour
# GCS not available in features.csv -> 2-component partial qSOFA (RR + SBP)
_qsofa_rr_col  = "resp_rate_last"
_qsofa_sbp_col = "sbp_last"
_has_qsofa = (_qsofa_rr_col in FEAT_COLS) and (_qsofa_sbp_col in FEAT_COLS)
if _has_qsofa:
    _qsofa_rr_idx  = FEAT_COLS.index(_qsofa_rr_col)
    _qsofa_sbp_idx = FEAT_COLS.index(_qsofa_sbp_col)
    print(f"    qSOFA: using {_qsofa_rr_col} (idx {_qsofa_rr_idx}) + {_qsofa_sbp_col} (idx {_qsofa_sbp_idx})")
else:
    print("    qSOFA: required columns missing, will skip qSOFA baseline")

# ---------------------------------------------------------------------------- #
#  3. BUILD SLIDING WINDOWS  ->  (N_windows, SEQ_LEN, N_FEAT)                  #
#     FIX 1: label = "will sepsis occur in the next HORIZON hours?"            #
# ---------------------------------------------------------------------------- #
print(f"\n[3] Building sliding windows (horizon={HORIZON}h) ...")
seq_X, seq_y, seq_g = [], [], []

for stay_id, grp in df.groupby("stay_id", sort=False):
    idx = grp.index.tolist()
    n   = len(idx)
    if n < SEQ_LEN:
        continue
    subj = df.loc[idx[0], group_key]
    for start in range(0, n - SEQ_LEN + 1, STRIDE):
        end = start + SEQ_LEN
        # FIXED: look HORIZON steps ahead from window end, not current step
        future_idx = idx[end : end + HORIZON]
        if len(future_idx) == 0:
            continue
        label = 1.0 if yfull[future_idx].sum() > 0 else 0.0
        seq_X.append(Xfull[idx[start:end]])   # (SEQ_LEN, N_FEAT)
        seq_y.append(label)
        seq_g.append(subj)

seq_X = np.array(seq_X, dtype=np.float32)
seq_y = np.array(seq_y, dtype=np.float32)
seq_g = np.array(seq_g)

# Optional cap
if args.max_windows is not None and len(seq_X) > args.max_windows:
    rng  = np.random.default_rng(42)
    keep = rng.choice(len(seq_X), args.max_windows, replace=False)
    keep.sort()
    seq_X, seq_y, seq_g = seq_X[keep], seq_y[keep], seq_g[keep]
    print(f"    Capped to {args.max_windows:,} windows (--max-windows)")

N_WIN = len(seq_X)
print(f"    Windows : {N_WIN:,}")
print(f"    Shape   : {seq_X.shape}")
print(f"    Positive: {seq_y.sum():.0f} ({seq_y.mean()*100:.2f}%)")

# Build qSOFA soft scores from LAST timestep of each window
# score = (RR>=22) + (SBP<=100), range 0-2, normalised to [0,1] as score/2
if _has_qsofa:
    qsofa_rr    = (seq_X[:, -1, _qsofa_rr_idx]  >= 22).astype(np.float32)
    qsofa_sbp   = (seq_X[:, -1, _qsofa_sbp_idx] <= 100).astype(np.float32)
    qsofa_score = (qsofa_rr + qsofa_sbp) / 2.0   # soft prob for AUROC
    print(f"    qSOFA: RR>=22 in {qsofa_rr.mean()*100:.1f}%  SBP<=100 in {qsofa_sbp.mean()*100:.1f}% of windows")

# ---------------------------------------------------------------------------- #
#  4. FIX 2: Temporal stats for flat models                                    #
#     Expand: (N, T, F) -> (N, T*F + 5*F) = raw sequence + mean/std/min/max/slope
# ---------------------------------------------------------------------------- #
def add_temporal_stats(X_3d):
    """Augment flattened sequences with per-feature temporal summary stats."""
    N, T, F = X_3d.shape
    means  = X_3d.mean(axis=1)                         # (N, F)
    stds   = X_3d.std(axis=1)                          # (N, F)
    mins   = X_3d.min(axis=1)                          # (N, F)
    maxs   = X_3d.max(axis=1)                          # (N, F)
    # Linear slope: project each feature's time series onto centered time axis
    t      = (np.arange(T, dtype=np.float32) - T / 2.0)
    denom  = float((t ** 2).sum())
    slopes = (X_3d * t[np.newaxis, :, np.newaxis]).sum(axis=1) / denom  # (N, F)
    raw    = X_3d.reshape(N, -1)                       # (N, T*F)
    return np.concatenate([raw, means, stds, mins, maxs, slopes], axis=1).astype(np.float32)

print(f"\n[4] Adding temporal stats: raw {N_WIN}x{SEQ_LEN*N_FEAT} -> enriched ...")
seq_X_flat_raw = seq_X.reshape(N_WIN, -1)             # raw flat, used only for LSTM scaler shape
# We compute the enriched flat version AFTER per-fold scaling (inside the fold loop)

# ---------------------------------------------------------------------------- #
#  5. LSTM MODEL with attention pooling                                         #
# ---------------------------------------------------------------------------- #
class SepsisDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


class SepsisLSTM(nn.Module):
    """2-layer LSTM with additive attention pooling."""
    def __init__(self, n_feat, hidden, n_layers, drop):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                            batch_first=True,
                            dropout=drop if n_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden, 1)
        self.norm  = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(drop)
        self.fc    = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)             # (B, T, hidden)
        attn_weights = torch.softmax(self.attention(out), dim=1) # (B, T, 1)
        context = torch.sum(attn_weights * out, dim=1)           # (B, hidden)
        context = self.norm(context)
        context = self.drop(context)
        return self.fc(context).squeeze(1)


# ---------------------------------------------------------------------------- #
#  6. METRIC HELPER                                                             #
# ---------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, model_name, prevalence=None):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    # Prevalence defaults to empirical positive rate if not provided
    prev  = prevalence if prevalence is not None else float(y_true.mean())
    auprc_lift = auprc / prev if prev > 0 else float("nan")  # ratio vs. random baseline

    # Primary: F1-optimal threshold (data-driven)
    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.01, 0.99, 198):
        f1 = f1_score(y_true, (y_prob >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    ybin = (y_prob >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, ybin).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    bar = "=" * 22
    print("\n" + bar + " " + model_name + " " + bar)
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  AUPRC      : {auprc:.4f}  (baseline={prev:.4f}, lift={auprc_lift:.1f}x over random)")
    print(f"  F1 (opt)   : {best_f1:.4f}  @  threshold={best_th:.3f}  [data-driven sweep]")
    print(f"  Sensitivity: {sens:.4f}  (@ F1-opt threshold)")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Precision  : {ppv:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Fixed operating points: clinically meaningful sensitivity targets
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_prob)
    print("  -- Fixed operating points --")
    for target_sens in [0.50, 0.70, 0.80]:
        idx = np.argmin(np.abs(tpr_arr - target_sens))
        th_f = float(thr_arr[idx])
        achieved_sens = float(tpr_arr[idx])
        achieved_spec = float(1.0 - fpr_arr[idx])
        print(f"    Sens~{target_sens:.0%}: Sens={achieved_sens:.3f}  Spec={achieved_spec:.3f}  threshold={th_f:.3f}")

    print("=" * (46 + len(model_name)))

    return dict(auroc=float(auroc), auprc=float(auprc),
                auprc_lift=round(auprc_lift, 2),
                prevalence=round(prev, 6),
                f1=float(best_f1), threshold=float(best_th),
                sensitivity=float(sens), specificity=float(spec),
                precision=float(ppv),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ---------------------------------------------------------------------------- #
#  7. GROUP-K-FOLD LOOP  (same folds for ALL models)                            #
# ---------------------------------------------------------------------------- #
print(f"\n[5] {N_FOLDS}-fold GroupKFold  (grouped by {group_key}) ...")
gkf    = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(seq_X, seq_y, groups=seq_g))

MODEL_NAMES = ["LSTM", "XGBoost", "RandomForest", "LogisticReg"]
oof = {name: np.zeros(N_WIN) for name in MODEL_NAMES}
fold_aurocs = {name: [] for name in MODEL_NAMES}

best_xgb_auroc = 0.0
best_xgb_model = None
best_imputer = None
best_scaler = None

SEP = "-" * 65

for fold, (tr_idx, va_idx) in enumerate(splits):
    print("\n" + SEP)
    print(f"  FOLD {fold+1}/{N_FOLDS}  |  train={len(tr_idx):,}  val={len(va_idx):,}"
          f"  |  pos_train={int(seq_y[tr_idx].sum())}  pos_val={int(seq_y[va_idx].sum())}")
    print(SEP)

    y_tr, y_va = seq_y[tr_idx], seq_y[va_idx]
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"  Positive weight (pos_w) : {pos_w:.1f}")

    # Impute missing values inside fold to prevent leakage
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(seq_X_flat_raw[tr_idx])
    X_va_imp = imputer.transform(seq_X_flat_raw[va_idx])

    # Per-fold scaler on raw flat (keeps LSTM 3-D shape correct)
    scaler       = StandardScaler()
    X_tr_raw     = scaler.fit_transform(X_tr_imp).astype(np.float32)
    X_va_raw     = scaler.transform(X_va_imp).astype(np.float32)

    # 3-D tensors for LSTM
    X_tr_3d = X_tr_raw.reshape(-1, SEQ_LEN, N_FEAT)
    X_va_3d = X_va_raw.reshape(-1, SEQ_LEN, N_FEAT)

    # FIX 2: enriched flat for classical models (temporal stats appended)
    X_tr_enr = add_temporal_stats(X_tr_3d)  # (N_tr, T*F + 5*F)
    X_va_enr = add_temporal_stats(X_va_3d)  # (N_va, T*F + 5*F)

    print(f"  Flat feature dim  : {X_tr_enr.shape[1]:,}  "
          f"(raw {SEQ_LEN*N_FEAT} + temporal stats {5*N_FEAT})")

    # ---- A) LSTM with attention -------------------------------------------
    print("  [LSTM] training ...")
    set_seed(42 + fold)
    model   = SepsisLSTM(N_FEAT, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    pos_w_t = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)
    optim   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    # ReduceLROnPlateau is more robust than cosine annealing on imbalanced data:
    # it only reduces LR when val AUROC stalls, not on a fixed schedule
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", patience=2, factor=0.5, min_lr=1e-6
    )

    tr_dl = DataLoader(SepsisDS(X_tr_3d, y_tr), batch_size=BATCH, shuffle=True,  num_workers=0)
    va_dl = DataLoader(SepsisDS(X_va_3d, y_va), batch_size=BATCH, shuffle=False, num_workers=0)

    best_auroc, best_preds, pat_cnt = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item() * len(yb)
        avg_loss = ep_loss / len(y_tr)

        model.eval()
        with torch.no_grad():
            preds = torch.cat([
                torch.sigmoid(model(xb.to(DEVICE))).cpu()
                for xb, _ in va_dl
            ]).numpy()
        ep_auc = roc_auc_score(y_va, preds)
        sched.step(ep_auc)   # ReduceLROnPlateau: step on val metric
        cur_lr = optim.param_groups[0]["lr"]
        print(f"    Epoch {ep+1:02d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Val AUROC: {ep_auc:.4f}  |  LR: {cur_lr:.2e}")

        if ep_auc > best_auroc + 1e-4:
            best_auroc, best_preds, pat_cnt = ep_auc, preds.copy(), 0
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print(f"    Early stop at epoch {ep+1}")
                break

    oof["LSTM"][va_idx] = best_preds
    fold_aurocs["LSTM"].append(best_auroc)
    print(f"  [LSTM]  fold AUROC: {best_auroc:.4f}")

    # ---- B) XGBoost (FIX 3: tuned params + early stopping) ---------------
    print("  [XGBoost] training ...")
    xgb_clf = XGBClassifier(
        n_estimators=500,        # was 200
        max_depth=6,             # was 5
        learning_rate=0.03,      # was 0.05
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=pos_w,
        eval_metric="aucpr",     # optimize AUPRC directly
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    xgb_clf.fit(
        X_tr_enr, y_tr,
        eval_set=[(X_va_enr, y_va)],
        verbose=False
    )
    p = xgb_clf.predict_proba(X_va_enr)[:, 1]
    oof["XGBoost"][va_idx] = p
    fa = roc_auc_score(y_va, p)
    fold_aurocs["XGBoost"].append(fa)
    if fa > best_xgb_auroc:
        best_xgb_auroc = fa
        # Store a copy of the model, or just keep reference since we create a new one each fold
        best_xgb_model = xgb_clf
        best_imputer = imputer
        best_scaler = scaler
    print(f"  [XGBoost] fold AUROC: {fa:.4f}  (best iter: {xgb_clf.best_iteration})")

    # ---- C) Random Forest --------------------------------------------------
    print("  [RandomForest] training ...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,        # was 200
        max_depth=10,            # was 8
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf_clf.fit(X_tr_enr, y_tr)
    p = rf_clf.predict_proba(X_va_enr)[:, 1]
    oof["RandomForest"][va_idx] = p
    fa = roc_auc_score(y_va, p)
    fold_aurocs["RandomForest"].append(fa)
    print(f"  [RandomForest] fold AUROC: {fa:.4f}")

    # ---- D) Logistic Regression (saga, converges on high-dim) -------------
    print("  [LogisticReg] training ...")
    lr_clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",           # scales to large/high-dim datasets
        random_state=42
    )
    lr_clf.fit(X_tr_enr, y_tr)
    p = lr_clf.predict_proba(X_va_enr)[:, 1]
    oof["LogisticReg"][va_idx] = p
    fa = roc_auc_score(y_va, p)
    fold_aurocs["LogisticReg"].append(fa)
    print(f"  [LogisticReg] fold AUROC: {fa:.4f}")

# ---------------------------------------------------------------------------- #
#  8. AGGREGATE OOF METRICS                                                     #
# ---------------------------------------------------------------------------- #
WIDE = "=" * 65
print("\n" + WIDE)
print("  UNIFIED BENCHMARK  --  Same sequences | Same features | Same folds")
print(f"  Seq-len={SEQ_LEN}h  |  Horizon={HORIZON}h  |  Stride={STRIDE}h  |  N_windows={N_WIN:,}")
print(WIDE)

WIN_PREVALENCE = float(seq_y.mean())
all_metrics = {}
for name in MODEL_NAMES:
    m = compute_metrics(seq_y, oof[name], name, prevalence=WIN_PREVALENCE)
    m["fold_aurocs"]     = [round(v, 4) for v in fold_aurocs[name]]
    m["fold_auroc_mean"] = round(float(np.mean(fold_aurocs[name])), 4)
    m["fold_auroc_std"]  = round(float(np.std(fold_aurocs[name])),  4)
    print(f"  Fold AUROCs: {m['fold_aurocs']}  ->  {m['fold_auroc_mean']:.4f} +/- {m['fold_auroc_std']:.4f}")
    all_metrics[name] = m

# qSOFA clinical baseline -- evaluated on SAME OOF windows (no training needed)
if _has_qsofa:
    print("\n[qSOFA] Evaluating clinical baseline on OOF windows ...")
    qsofa_oof = np.zeros(N_WIN)
    qsofa_fold_aurocs = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        y_va = seq_y[va_idx]
        q_va = qsofa_score[va_idx]
        fa   = roc_auc_score(y_va, q_va)
        qsofa_fold_aurocs.append(fa)
        qsofa_oof[va_idx] = q_va
        print(f"  Fold {fold+1}: qSOFA AUROC = {fa:.4f}")

    # (A) Max-F1 swept threshold -- comparable to ML models but NOT clinical convention
    q_m = compute_metrics(seq_y, qsofa_oof, "qSOFA (RR+SBP) swept-th", prevalence=WIN_PREVALENCE)
    q_m["fold_aurocs"]     = [round(v, 4) for v in qsofa_fold_aurocs]
    q_m["fold_auroc_mean"] = round(float(np.mean(qsofa_fold_aurocs)), 4)
    q_m["fold_auroc_std"]  = round(float(np.std(qsofa_fold_aurocs)),  4)
    q_m["threshold_type"]  = "max-F1 sweep (not clinical convention)"
    q_m["note"] = ("2-component qSOFA: RR>=22 + SBP<=100. GCS unavailable in features.csv. "
                   "Threshold chosen by max-F1 sweep, same as ML models.")
    print(f"  Fold AUROCs: {q_m['fold_aurocs']}  ->  {q_m['fold_auroc_mean']:.4f} +/- {q_m['fold_auroc_std']:.4f}")
    all_metrics["qSOFA (swept)"] = q_m

    # (B) Standard clinical threshold: score = 1.0 (BOTH criteria met)
    # Equivalent to qSOFA >= 2/2 (since GCS is missing, we require both RR and SBP criteria)
    clinical_th = 1.0  # score/2 = 1.0 means both RR>=22 AND SBP<=100
    q_bin_clin  = (qsofa_oof >= clinical_th).astype(int)
    if q_bin_clin.sum() > 0 and (1 - q_bin_clin).sum() > 0:
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(seq_y, q_bin_clin).ravel()
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
        ppv_c  = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        f1_c   = f1_score(seq_y, q_bin_clin, zero_division=0)
        print("\n====== qSOFA (RR+SBP) clinical threshold (both criteria met) ======")
        print(f"  Clinical threshold : score = 1.0 (RR>=22 AND SBP<=100)")
        print(f"  Sensitivity        : {sens_c:.4f}")
        print(f"  Specificity        : {spec_c:.4f}")
        print(f"  Precision          : {ppv_c:.4f}")
        print(f"  F1                 : {f1_c:.4f}")
        print(f"  TP={tp_c}  FP={fp_c}  FN={fn_c}  TN={tn_c}")
        print("===================================================================")
        clin_m = dict(
            auroc=q_m["auroc"], auprc=q_m["auprc"],
            auprc_lift=q_m["auprc_lift"], prevalence=q_m["prevalence"],
            threshold=float(clinical_th), threshold_type="clinical convention (both criteria met)",
            sensitivity=float(sens_c), specificity=float(spec_c),
            precision=float(ppv_c), f1=float(f1_c),
            tp=int(tp_c), fp=int(fp_c), fn=int(fn_c), tn=int(tn_c),
            fold_aurocs=q_m["fold_aurocs"],
            fold_auroc_mean=q_m["fold_auroc_mean"],
            fold_auroc_std=q_m["fold_auroc_std"],
            note=("2-component qSOFA. Clinical threshold = both RR>=22 AND SBP<=100. "
                  "GCS unavailable in features.csv (standard qSOFA uses 3 components).")
        )
        all_metrics["qSOFA (clinical)"] = clin_m

# ---------------------------------------------------------------------------- #
#  8.5 SAVE OOF PREDICTIONS (FOR BOOTSTRAP CI)                                  #
# ---------------------------------------------------------------------------- #
if args.save_oof:
    out_npz = os.path.join(REPORTS_DIR, "oof_predictions.npz")
    save_dict = {"seq_y": seq_y}
    for name in MODEL_NAMES:
        save_dict[name] = oof[name]
    if _has_qsofa:
        save_dict["qSOFA (swept)"] = qsofa_oof
        save_dict["qSOFA (clinical)"] = qsofa_oof
    np.savez_compressed(out_npz, **save_dict)
    print(f"\nSaved OOF predictions for bootstrap CI: {out_npz}")

# ---------------------------------------------------------------------------- #
#  9. SAVE RESULTS JSON                                                         #
# ---------------------------------------------------------------------------- #
enriched_dim = SEQ_LEN * N_FEAT + 5 * N_FEAT
out_json = os.path.join(REPORTS_DIR, "unified_benchmark_results.json")
with open(out_json, "w") as f:
    json.dump({
        "config": {
            "seq_len": SEQ_LEN, "stride": STRIDE, "horizon": HORIZON,
            "n_folds": N_FOLDS, "lstm_epochs": EPOCHS,
            "n_windows": N_WIN, "n_features_raw": N_FEAT,
            "n_features_enriched_flat": enriched_dim,
            "label": f"sepsis onset in next {HORIZON} hours",
            "split": f"{N_FOLDS}-fold GroupKFold by {group_key}",
            "improvements": [
                f"label fixed to {HORIZON}h prediction horizon",
                "temporal stats added (mean/std/min/max/slope per feature)",
                "XGBoost: 500 trees, lr=0.03, aucpr eval, early stopping",
                "LSTM: attention pooling, AdamW, cosine LR schedule",
                "RF: 300 trees, max_depth=10",
                "LR: saga solver, max_iter=5000"
            ]
        },
        "results": all_metrics
    }, f, indent=2)
print(f"\nSaved: {out_json}")

if best_xgb_model is not None:
    best_xgb_path = os.path.join(PROCESSED_DIR, "unified_xgb_model.json")
    best_xgb_model.save_model(best_xgb_path)
    import joblib
    joblib.dump(best_imputer, os.path.join(PROCESSED_DIR, "unified_imputer.pkl"))
    joblib.dump(best_scaler, os.path.join(PROCESSED_DIR, "unified_scaler.pkl"))
    print(f"Saved best XGBoost model (AUROC {best_xgb_auroc:.4f}) to {best_xgb_path}")

# ---------------------------------------------------------------------------- #
#  10. SUMMARY TABLE                                                            #
# ---------------------------------------------------------------------------- #
print("\n" + "=" * 90)
print(f"  {'Model':<22} {'AUROC':>7} {'AUPRC':>7} {'AUPRC-lift':>10} {'Sens':>7} {'Spec':>7} {'F1':>7}")
print("-" * 90)
for name, m in all_metrics.items():
    lift_str = f"{m.get('auprc_lift', float('nan')):>9.1f}x"
    print(f"  {name:<22} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {lift_str} "
          f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f} {m['f1']:>7.4f}")
print("=" * 90)
print(f"  Label           : sepsis onset in next {HORIZON} hours (prediction horizon)")
print(f"  Seq-len         : {SEQ_LEN}h lookback window")
print(f"  Window prevalence: {WIN_PREVALENCE*100:.2f}%  (AUPRC baseline = {WIN_PREVALENCE:.4f})")
print(f"  Feature set     : {N_FEAT} raw + temporal stats = {enriched_dim} total (flat models)")
if args.no_sofa:
    print(f"  SOFA features   : EXCLUDED ({len(SOFA_COLS)} cols removed -- leakage ablation)")
else:
    print(f"  SOFA features   : INCLUDED ({len(SOFA_COLS)} cols -- see ablation for leakage estimate)")
print(f"  Fold structure  : {N_FOLDS}-fold GroupKFold (same splits for all models)")
print(f"  Sens/Spec above : @ F1-optimal threshold (data-driven). See fixed-OP table above.")
print("=" * 90)

# ---------------------------------------------------------------------------- #
#  11. PLOTS                                                                    #
# ---------------------------------------------------------------------------- #
if args.plot:
    print("\nGenerating plots ...")
    COLORS = {
        "LSTM":             "#2563EB",
        "XGBoost":          "#D97706",
        "RandomForest":     "#16A34A",
        "LogisticReg":      "#DC2626",
        "qSOFA (RR+SBP)":   "#7C3AED",
    }
    STYLES = {
        "LSTM":             "-",
        "XGBoost":          "--",
        "RandomForest":     "-.",
        "LogisticReg":      ":",
        "qSOFA (RR+SBP)":   (0, (3, 1, 1, 1)),  # dash-dot-dot
    }

    for name, preds in oof.items():
        # Individual ROC + PR
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{name} Performance  --  {SEQ_LEN}h seq | {HORIZON}h horizon | {N_FOLDS}-fold", fontsize=12, fontweight="bold")
        
        fpr, tpr, _ = roc_curve(seq_y, preds)
        axes[0].plot(fpr, tpr, color=COLORS.get(name, "#333333"), lw=2,
                     label=f"AUROC = {all_metrics[name]['auroc']:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        axes[0].set_xlabel("False Positive Rate", fontsize=11)
        axes[0].set_ylabel("True Positive Rate", fontsize=11)
        axes[0].set_title(f"ROC Curve - {name}", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9, loc="lower right")
        axes[0].grid(alpha=0.3)

        prec, rec, _ = precision_recall_curve(seq_y, preds)
        axes[1].plot(rec, prec, color=COLORS.get(name, "#333333"), lw=2,
                     label=f"AUPRC = {all_metrics[name]['auprc']:.4f}")
        axes[1].axhline(seq_y.mean(), color="gray", ls="--", lw=1,
                        label=f"Prevalence ({seq_y.mean():.3f})")
        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].set_title(f"Precision-Recall Curve - {name}", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=9, loc="upper right")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        out = os.path.join(REPORTS_DIR, f"{safe_name}_roc_pr.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {safe_name}_roc_pr.png")
        plt.close()

    # Per-fold AUROC grouped bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(N_FOLDS)
    w     = 0.18
    names = list(oof.keys())
    for i, name in enumerate(names):
        bars = ax.bar(x + i*w, fold_aurocs[name], w, label=name,
                      color=COLORS[name], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, fold_aurocs[name]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Per-Fold AUROC  --  All Models, Same Splits", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "unified_benchmark_fold_aurocs.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  Saved: unified_benchmark_fold_aurocs.png")
    plt.close()

    # AUROC comparison bar chart (OOF overall)
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(all_metrics.keys())
    aurocs = [all_metrics[n]["auroc"] for n in model_names]
    auprcs = [all_metrics[n]["auprc"] for n in model_names]
    xi = np.arange(len(model_names))
    b1 = ax.bar(xi - 0.2, aurocs, 0.35, label="AUROC",
                color=[COLORS.get(n, "#888888") for n in model_names], alpha=0.85)
    b2 = ax.bar(xi + 0.2, auprcs, 0.35, label="AUPRC",
                color=[COLORS.get(n, "#888888") for n in model_names], alpha=0.4,
                edgecolor=[COLORS.get(n, "#888888") for n in model_names], linewidth=1.5)
    for bar, v in zip(b1, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, auprcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xi)
    ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"AUROC & AUPRC -- {SEQ_LEN}h window, {HORIZON}h horizon",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "unified_benchmark_auroc_auprc.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  Saved: unified_benchmark_auroc_auprc.png")
    plt.close()

print("\nDone.")
