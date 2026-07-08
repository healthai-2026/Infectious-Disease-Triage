"""
train_lstm.py
-------------
True time-sequential LSTM for sepsis prediction.

Data flow (no preprocess.py needed):
  chartevents.csv.gz  -> hourly vital-sign grids per ICU stay
  icustays.csv.gz     -> stay metadata (intime, subject_id, hadm_id)
  patients.csv.gz     -> age, gender
  diagnoses_icd.csv.gz + d_icd_diagnoses.csv.gz -> ICD-based sepsis label

Each ICU stay becomes a (T x 7) time-series (T hourly steps, 7 features).
Sliding windows of length SEQ_LEN are fed to a 2-layer LSTM.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, f1_score, roc_curve
)

# ---------------------------------------------------------------------------- #
#  Paths & Config                                                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train a lightweight LSTM sepsis predictor")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per fold")
parser.add_argument("--n-folds", type=int, default=2, help="Number of GroupKFold folds")
parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
parser.add_argument("--seq-len", type=int, default=12, help="Hours per input window")
parser.add_argument("--max-hours", type=int, default=24, help="How far into each stay to look")
parser.add_argument("--window-stride", type=int, default=3, help="Stride between sliding windows")
parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on generated training sequences; omit to use all windows")
parser.add_argument("--plot", action="store_true", help="Generate performance plots")
args = parser.parse_args()

DATA_DIR      = r"C:\PS1\Infectious-Disease-Triage\Data\mimic-iv-3.1"
PROCESSED_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\processed"
REPORTS_DIR   = r"C:\PS1\Infectious-Disease-Triage\reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SEQ_LEN      = args.seq_len
MAX_HOURS    = args.max_hours
BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
N_FOLDS      = args.n_folds
PATIENCE     = 5
MAX_SEQUENCES = args.max_sequences
WINDOW_STRIDE = args.window_stride
GENERATE_PLOTS = args.plot
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Training config: epochs={EPOCHS}, folds={N_FOLDS}, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, max_hours={MAX_HOURS}, stride={WINDOW_STRIDE}, max_sequences={'all' if MAX_SEQUENCES is None else MAX_SEQUENCES}")

# ---------------------------------------------------------------------------- #
#  Step 1 & 2 & 3 — Load preprocessed features.csv & Build sequences            #
# ---------------------------------------------------------------------------- #
print("\n[1/3] Loading features.csv...")
features_file = os.path.join(PROCESSED_DIR, "features.csv")
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file not found at {features_file}. Please run preprocess.py first.")

df = pd.read_csv(features_file)
print(f"Loaded dataset with shape: {df.shape}")

# Drop rows where target is missing
df = df.dropna(subset=['label'])

print("\n[2/3] Building hourly grids and sequences from features.csv...")
grouped = df.groupby('stay_id')

N_VITALS   = 5
N_FEATURES = 7    # 5 vitals + age + gender

sequences = []
labels_out = []
groups_out = []
skipped = 0

# Extract group data into a list of tuples to speed up grouping loop
for sid, group in grouped:
    if len(group) == 0:
        skipped += 1
        continue
        
    subject_id = group['subject_id'].iloc[0]
    label = group['label'].iloc[0]
    age = group['age'].iloc[0]
    gender = group['gender'].iloc[0]
    
    # Build hourly grid of shape (MAX_HOURS, N_VITALS)
    grid = np.full((MAX_HOURS, N_VITALS), np.nan, dtype=np.float32)
    
    for row in group.itertuples():
        h = int(row.hours_since_admit)
        if 0 <= h < MAX_HOURS:
            grid[h, 0] = row.heart_rate_mean
            grid[h, 1] = row.mbp_mean
            grid[h, 2] = row.resp_rate_mean
            grid[h, 3] = row.spo2_mean
            grid[h, 4] = row.temp_mean
            
    # Forward-fill / default-fill
    defaults = [80.0, 80.0, 15.0, 98.0, 37.0]
    for vi in range(N_VITALS):
        col = grid[:, vi]
        last = defaults[vi]
        for h in range(MAX_HOURS):
            if not np.isnan(col[h]):
                last = col[h]
            else:
                col[h] = last
        grid[:, vi] = col
        
    # Append age and gender
    age_col = np.full((MAX_HOURS, 1), age, dtype=np.float32)
    gender_col = np.full((MAX_HOURS, 1), gender, dtype=np.float32)
    grid_full = np.concatenate([grid, age_col, gender_col], axis=1)
    
    # Sliding windows
    for start in range(0, MAX_HOURS - SEQ_LEN + 1, WINDOW_STRIDE):
        seq = grid_full[start : start + SEQ_LEN]
        sequences.append(seq)
        labels_out.append(label)
        groups_out.append(subject_id)
        if MAX_SEQUENCES is not None and len(sequences) >= MAX_SEQUENCES:
            break
    if MAX_SEQUENCES is not None and len(sequences) >= MAX_SEQUENCES:
        break

sequences  = np.array(sequences,  dtype=np.float32)  # (N, SEQ_LEN, 7)
labels_out = np.array(labels_out, dtype=np.float32)
groups_out = np.array(groups_out)

print(f"  Skipped stays: {skipped}")
print(f"  Total sequences : {sequences.shape[0]:,}")
print(f"  Sequence shape  : {sequences.shape}")
print(f"  Positive (sepsis): {labels_out.sum():.0f} ({labels_out.mean()*100:.2f}%)")

# ---------------------------------------------------------------------------- #
#  Dataset & Model                                                              #
# ---------------------------------------------------------------------------- #
class SepsisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]


class SepsisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # use last timestep hidden state
        return self.fc(self.dropout(out)).squeeze(-1)


# ---------------------------------------------------------------------------- #
#  Step 4 — 5-fold GroupKFold Cross-Validation                                 #
# ---------------------------------------------------------------------------- #
print(f"\n[4/4] Running {N_FOLDS}-fold GroupKFold CV...")

gkf        = GroupKFold(n_splits=N_FOLDS)
oof_preds  = np.zeros(len(sequences))
fold_aurocs = []
last_losses, last_val_aurocs = [], []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(sequences, labels_out, groups=groups_out)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    X_tr, y_tr = sequences[tr_idx], labels_out[tr_idx]
    X_va, y_va = sequences[va_idx], labels_out[va_idx]

    # Normalize per-feature using train stats
    scaler = StandardScaler()
    ntr, sl, nf = X_tr.shape
    X_tr = scaler.fit_transform(X_tr.reshape(-1, nf)).reshape(ntr, sl, nf).astype(np.float32)
    nva  = X_va.shape[0]
    X_va = scaler.transform(X_va.reshape(-1, nf)).reshape(nva, sl, nf).astype(np.float32)

    pos_w = torch.tensor(
        [(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)],
        dtype=torch.float32
    ).to(DEVICE)

    tr_loader = DataLoader(
        SepsisDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    va_loader = DataLoader(
        SepsisDataset(X_va, y_va),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model     = SepsisLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_auroc = 0.0
    best_preds = None
    ep_losses, ep_aurocs = [], []
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(yb)

        avg_loss = ep_loss / len(y_tr)
        ep_losses.append(avg_loss)

        model.eval()
        preds_ep = []
        with torch.no_grad():
            for xb, _ in va_loader:
                preds_ep.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        preds_ep = np.array(preds_ep)
        ep_auroc = roc_auc_score(y_va, preds_ep)
        ep_aurocs.append(ep_auroc)
        scheduler.step(1 - ep_auroc)

        if ep_auroc > best_auroc + 1e-4:
            best_auroc = ep_auroc
            best_preds = preds_ep.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}/{EPOCHS}.")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUROC: {ep_auroc:.4f}")

    oof_preds[va_idx] = best_preds
    fold_aurocs.append(best_auroc)
    print(f"  Best AUROC: {best_auroc:.4f}")

    if fold == N_FOLDS - 1:
        last_losses, last_val_aurocs = ep_losses, ep_aurocs

# ---------------------------------------------------------------------------- #
#  Metrics                                                                      #
# ---------------------------------------------------------------------------- #
lstm_auroc = roc_auc_score(labels_out, oof_preds)
lstm_prec, lstm_rec, _ = precision_recall_curve(labels_out, oof_preds)
lstm_auprc = auc(lstm_rec, lstm_prec)

best_f1, best_thresh = 0.0, 0.5
for th in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(labels_out, (oof_preds >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, th

preds_bin = (oof_preds >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(labels_out, preds_bin).ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0

print("\n" + "="*20 + " LSTM RESULTS " + "="*20)
print(f"LSTM - True hourly time-series (SEQ_LEN={SEQ_LEN}h, threshold={best_thresh:.3f}):")
print(f"  AUROC:       {lstm_auroc:.4f}")
print(f"  AUPRC:       {lstm_auprc:.4f}")
print(f"  Sensitivity: {sens:.4f}")
print(f"  Specificity: {spec:.4f}")
print(f"  Precision:   {ppv:.4f}")
print(f"  F1-Score:    {best_f1:.4f}")
print(f"  Confusion:   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Fold AUROCs: {[f'{v:.4f}' for v in fold_aurocs]}")
print(f"  Mean +/- Std: {np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")
print("=" * 54)

# ---------------------------------------------------------------------------- #
#  Visualizations                                                               #
# ---------------------------------------------------------------------------- #
if GENERATE_PLOTS:
    print("\nGenerating visualizations...")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'LSTM (True Hourly Time-Series, SEQ_LEN={SEQ_LEN}h)\nSepsis Prediction Performance',
                 fontsize=13, fontweight='bold', y=1.02)

    # ROC
    fpr, tpr, _ = roc_curve(labels_out, oof_preds)
    axes[0, 0].plot(fpr, tpr, color='royalblue', linewidth=2, label=f'LSTM (AUROC={lstm_auroc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10); axes[0, 0].grid(alpha=0.3)

    # PR curve
    axes[0, 1].plot(lstm_rec, lstm_prec, color='darkorange', linewidth=2,
                    label=f'LSTM (AUPRC={lstm_auprc:.4f})')
    axes[0, 1].axhline(labels_out.mean(), color='gray', linestyle='--', linewidth=1,
                       label=f'Prevalence ({labels_out.mean():.3f})')
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10); axes[0, 1].grid(alpha=0.3)

    # Confusion matrix
    cm = confusion_matrix(labels_out, preds_bin)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False,
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # Training curve (last fold)
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(range(1, EPOCHS+1), last_losses,     color='royalblue',  linewidth=2, label='Train Loss')
    ax2.plot(range(1, EPOCHS+1), last_val_aurocs, color='darkorange', linewidth=2,
             linestyle='--', label='Val AUROC')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss',  color='royalblue',  fontsize=11)
    ax2.set_ylabel('Val AUROC',   color='darkorange', fontsize=11)
    axes[1, 1].set_title(f'Training Curve (Fold {N_FOLDS})', fontsize=12, fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10); ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "lstm_performance.png"), dpi=300, bbox_inches='tight')
    print("Saved: lstm_performance.png")
    plt.close()

    # Per-fold AUROC bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f'Fold {i+1}' for i in range(N_FOLDS)], fold_aurocs,
                  color='royalblue', alpha=0.8, edgecolor='navy')
    ax.axhline(np.mean(fold_aurocs), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(fold_aurocs):.4f}')
    for bar, val in zip(bars, fold_aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('Fold', fontsize=11); ax.set_ylabel('AUROC', fontsize=11)
    ax.set_title('LSTM Per-Fold AUROC (GroupKFold CV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "lstm_fold_aurocs.png"), dpi=300, bbox_inches='tight')
    print("Saved: lstm_fold_aurocs.png")
    plt.close()
else:
    print("Skipping plot generation. Add --plot to create them.")

# ---------------------------------------------------------------------------- #
#  Train final model & save                                                     #
# ---------------------------------------------------------------------------- #
print("\nTraining final LSTM on all data...")
scaler_final = StandardScaler()
n_all, sl, nf = sequences.shape
X_all = scaler_final.fit_transform(
    sequences.reshape(-1, nf)
).reshape(n_all, sl, nf).astype(np.float32)

pos_w_f = torch.tensor(
    [(len(labels_out) - labels_out.sum()) / max(labels_out.sum(), 1)],
    dtype=torch.float32
).to(DEVICE)

final_loader = DataLoader(SepsisDataset(X_all, labels_out), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
final_model  = SepsisLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
final_optim  = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
final_crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w_f)

for epoch in range(EPOCHS):
    final_model.train()
    for xb, yb in final_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        final_optim.zero_grad()
        loss = final_crit(final_model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        final_optim.step()

model_path = os.path.join(PROCESSED_DIR, "sepsis_lstm_model.pt")
torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler_mean':  scaler_final.mean_,
    'scaler_scale': scaler_final.scale_,
    'vital_ids':    ["heart_rate_mean", "mbp_mean", "resp_rate_mean", "spo2_mean", "temp_mean"],
    'hyperparams': {
        'input_size':  N_FEATURES,
        'hidden_size': HIDDEN_SIZE,
        'num_layers':  NUM_LAYERS,
        'dropout':     DROPOUT,
        'seq_len':     SEQ_LEN,
    }
}, model_path)
print(f"Saved final model -> {model_path}")
print("Training completed successfully!")
