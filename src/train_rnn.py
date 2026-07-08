import pandas as pd
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configurations
NPZ_PATH = r"C:\PS1\Infectious-Disease-Triage\Data\processed\sepsis_sequential.npz"
CSV_PATH = r"C:\PS1\Infectious-Disease-Triage\Data\processed\features.csv"
OUTPUT_DIR = r"C:\PS1\Infectious-Disease-Triage\reports"

# PyTorch LSTM Model Definition
class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # Linear output for binary logit
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # Select the output from the last time step (hour 24)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        logit = self.fc(last_out)
        return logit

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

def ffill_3d(X):
    """
    Perform forward-fill stay-by-stay along the time axis (axis 1).
    X shape: (num_stays, 24, num_features)
    """
    N, T, F = X.shape
    X_filled = X.copy()
    for i in range(N):
        for f in range(F):
            last_val = np.nan
            for t in range(T):
                if np.isnan(X_filled[i, t, f]):
                    X_filled[i, t, f] = last_val
                else:
                    last_val = X_filled[i, t, f]
    return X_filled


def build_fallback_sequences(csv_path):
    print(f"{NPZ_PATH} not found; building fallback inputs from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['label']).copy()

    seq_candidates = [
        'heart_rate_mean', 'sbp_mean', 'dbp_mean', 'mbp_mean', 'resp_rate_mean',
        'temp_mean', 'spo2_mean', 'glucose_mean', 'sodium_mean', 'creatinine_mean', 'wbc_mean',
        'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temp', 'spo2', 'glucose', 'sodium', 'creatinine', 'wbc'
    ]
    selected_seq_cols = [col for col in seq_candidates if col in df.columns]

    if not selected_seq_cols:
        raise ValueError(f"No compatible sequence columns found in {csv_path}")

    for col in selected_seq_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(['stay_id', 'hours_since_admit'])
    grouped = df.groupby('stay_id', sort=False)

    stay_ids = []
    labels = []
    seq_rows = []
    static_rows = []

    for sid, group in grouped:
        if group.empty:
            continue

        seq = group[selected_seq_cols].copy()
        seq = seq.ffill().bfill()
        seq = seq.iloc[:24, :].to_numpy(dtype=np.float32)

        if seq.shape[0] < 24:
            pad = np.repeat(seq[-1:].astype(np.float32), 24 - seq.shape[0], axis=0)
            seq = np.vstack([seq, pad])

        if seq.shape[1] < 11:
            seq = np.pad(seq, ((0, 0), (0, 11 - seq.shape[1])), mode='constant')

        stay_ids.append(int(sid))
        labels.append(int(group['label'].iloc[0]))
        seq_rows.append(seq[:24, :11])

        age_val = group['age'].iloc[0] if 'age' in group.columns else 0.0
        gender_val = group['gender'].iloc[0] if 'gender' in group.columns else 0.0
        if isinstance(gender_val, str):
            gender_val = 1.0 if gender_val.lower() in {'m', 'male', '1', 'true'} else 0.0
        else:
            gender_val = float(gender_val)

        age_val = float(age_val) if pd.notna(age_val) else 0.0
        static_rows.append(np.array([age_val, gender_val], dtype=np.float32))

    X_seq = np.stack(seq_rows, axis=0)
    X_static = np.stack(static_rows, axis=0)
    y = np.array(labels, dtype=np.int64)
    stay_ids = np.array(stay_ids, dtype=np.int64)

    print(f"Built fallback sequential data shape: {X_seq.shape}")
    print(f"Built fallback static data shape: {X_static.shape}")
    return X_seq, X_static, y, stay_ids


def main():
    start_time = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Step 1: Loading sequential dataset...")
    if os.path.exists(NPZ_PATH):
        data = np.load(NPZ_PATH)
        X_seq = data['X_seq']
        X_static = data['X_static']
        y = data['y']
        stay_ids = data['stay_ids']
        print(f"Loaded sequential data shape: {X_seq.shape}")
        print(f"Loaded static data shape: {X_static.shape}")
    else:
        X_seq, X_static, y, stay_ids = build_fallback_sequences(CSV_PATH)
        print(f"Loaded fallback sequential data shape: {X_seq.shape}")
        print(f"Loaded fallback static data shape: {X_static.shape}")
    
    # 1. Patient-safe split for train/test sets (80% train, 20% test)
    indices = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(indices, y, groups=stay_ids))
    
    X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
    X_static_train, X_static_test = X_static[train_idx], X_static[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 2. Sequential Imputation (Forward Fill)
    print("\nStep 2: Imputing clinical sequences using forward-fill...")
    X_seq_train_ffill = ffill_3d(X_seq_train)
    X_seq_test_ffill = ffill_3d(X_seq_test)
    
    # 3. Training Median Imputation for remaining NaNs (e.g. features never measured in a stay)
    # Compute median of each clinical feature on the training set (flattened)
    medians = np.nanmedian(X_seq_train_ffill.reshape(-1, 11), axis=0)
    print(f"Computed clinical features medians for remaining missing values:\n{medians}")
    
    for f in range(11):
        # If median itself is nan (extremely unlikely, but for safety), fill with 0
        fill_val = medians[f] if not np.isnan(medians[f]) else 0.0
        X_seq_train_ffill[np.isnan(X_seq_train_ffill[:, :, f]), f] = fill_val
        X_seq_test_ffill[np.isnan(X_seq_test_ffill[:, :, f]), f] = fill_val
        
    assert not np.isnan(X_seq_train_ffill).any(), "NaNs still remain in X_seq_train!"
    assert not np.isnan(X_seq_test_ffill).any(), "NaNs still remain in X_seq_test!"
    
    # 4. Concatenate Static demographics along the time dimension
    # Duplicate static features across the 24 hours: shape (N, 24, 13)
    X_static_seq_train = np.repeat(X_static_train[:, np.newaxis, :], 24, axis=1)
    X_static_seq_test = np.repeat(X_static_test[:, np.newaxis, :], 24, axis=1)
    
    # Concatenate sequence + static features along feature dimension: shape (N, 24, 24)
    X_train_full = np.concatenate([X_seq_train_ffill, X_static_seq_train], axis=2)
    X_test_full = np.concatenate([X_seq_test_ffill, X_static_seq_test], axis=2)
    
    print(f"Final RNN train features shape: {X_train_full.shape}")
    print(f"Final RNN test features shape: {X_test_full.shape}")
    
    # 5. Standard Scaling
    N_tr, T_tr, F_tr = X_train_full.shape
    scaler = StandardScaler()
    X_train_flat = X_train_full.reshape(-1, F_tr)
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(N_tr, T_tr, F_tr)
    
    N_te, T_te, F_te = X_test_full.shape
    X_test_flat = X_test_full.reshape(-1, F_te)
    X_test_scaled = scaler.transform(X_test_flat).reshape(N_te, T_te, F_te)
    
    print("\nStep 3: Building and training PyTorch LSTM model...")
    # Convert numpy arrays to Torch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Prepare DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Calculate scale weight for BCE loss to handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32)
    
    # Initialize LSTM
    model = SepsisLSTM(input_dim=F_tr, hidden_dim=64, num_layers=1, dropout=0.2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    # Train Loop
    epochs = 20
    model.train()
    print("Training LSTM model...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        avg_loss = epoch_loss / len(train_dataset)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {avg_loss:.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        rnn_prob = torch.sigmoid(test_logits).numpy().flatten()
        rnn_pred = (rnn_prob >= 0.5).astype(int)
        
    print("\n=== RNN (LSTM) Performance ===")
    print(classification_report(y_test, rnn_pred))
    
    rnn_fpr, rnn_tpr, _ = roc_curve(y_test, rnn_prob)
    rnn_auc = auc(rnn_fpr, rnn_tpr)
    rnn_precision, rnn_recall, _ = precision_recall_curve(y_test, rnn_prob)
    rnn_pr_auc = average_precision_score(y_test, rnn_prob)
    print(f"RNN ROC-AUC: {rnn_auc:.4f} | PR-AUC: {rnn_pr_auc:.4f}")
    
    rnn_cm = confusion_matrix(y_test, rnn_pred)
    torch.save(
        {'model_state_dict': model.state_dict(), 'input_dim': F_tr},
        os.path.join(OUTPUT_DIR, 'rnn_model.pt')
    )
    print(f"Saved RNN model to {os.path.join(OUTPUT_DIR, 'rnn_model.pt')}")
    
    print("\nStep 4: Training and evaluating RF and XGBoost baselines on the same split...")
    # Load static features dataset to train RF and XGBoost on richer stay-level features
    df_static = pd.read_csv(CSV_PATH)
    exclude_cols = {'stay_id', 'subject_id', 'hadm_id', 'time', 'label', 'hours_since_admit'}
    stay_level_features = [
        col for col in df_static.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_static[col])
    ]
    df_stay_level = (
        df_static.groupby('stay_id', as_index=False)
        .agg({'label': 'max', 'age': 'first', 'gender': 'first', **{col: 'mean' for col in stay_level_features if col not in {'age', 'gender', 'label'}}})
    )
    df_stay_level = df_stay_level.set_index('stay_id').reindex(stay_ids).reset_index()
    
    y_static = df_stay_level['label'].astype(int)
    X_static_all = df_stay_level[[col for col in df_stay_level.columns if col not in {'stay_id', 'label'}]].copy()
    X_static_all['gender'] = pd.to_numeric(X_static_all['gender'], errors='coerce').fillna(0).astype(int)
    X_static_all['age'] = pd.to_numeric(X_static_all['age'], errors='coerce').fillna(X_static_all['age'].median())
    
    # Split using same indices
    X_s_train, X_s_test = X_static_all.iloc[train_idx].copy(), X_static_all.iloc[test_idx].copy()
    y_s_train, y_s_test = y_static.iloc[train_idx], y_static.iloc[test_idx]
    
    # Verify labels match
    assert np.array_equal(y_s_train.values, y_train), "Labels mismatch in train split!"
    assert np.array_equal(y_s_test.values, y_test), "Labels mismatch in test split!"
    
    # Median Imputer
    static_imputer = SimpleImputer(strategy='median')
    X_s_train_imp = pd.DataFrame(static_imputer.fit_transform(X_s_train), columns=X_s_train.columns, index=X_s_train.index)
    X_s_test_imp = pd.DataFrame(static_imputer.transform(X_s_test), columns=X_s_test.columns, index=X_s_test.index)
    
    # 1. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_s_train_imp, y_s_train)
    rf_pred = rf.predict(X_s_test_imp)
    rf_prob = rf.predict_proba(X_s_test_imp)[:, 1]
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    rf_auc = auc(rf_fpr, rf_tpr)
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
    rf_pr_auc = average_precision_score(y_test, rf_prob)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    # 2. XGBoost
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    xgb.fit(X_s_train_imp, y_s_train)
    xgb_pred = xgb.predict(X_s_test_imp)
    xgb_prob = xgb.predict_proba(X_s_test_imp)[:, 1]
    
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_prob)
    xgb_pr_auc = average_precision_score(y_test, xgb_prob)
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    
    print(f"Random Forest ROC-AUC: {rf_auc:.4f} | PR-AUC: {rf_pr_auc:.4f}")
    print(f"XGBoost ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_pr_auc:.4f}")
    
    print("\nStep 5: Generating comparison plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Comparative ROC & PR Curves (All 3 Models)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # ROC Curves
    axes[0].plot(rf_fpr, rf_tpr, color='#1f77b4', lw=2.5, label=f'Random Forest (AUC = {rf_auc:.3f})')
    axes[0].plot(xgb_fpr, xgb_tpr, color='#ff7f0e', lw=2.5, label=f'XGBoost (AUC = {xgb_auc:.3f})')
    axes[0].plot(rnn_fpr, rnn_tpr, color='#2ca02c', lw=2.5, label=f'RNN - LSTM (AUC = {rnn_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", frameon=True)
    
    # PR Curves
    axes[1].plot(rf_recall, rf_precision, color='#1f77b4', lw=2.5, label=f'Random Forest (AP = {rf_pr_auc:.3f})')
    axes[1].plot(xgb_recall, xgb_precision, color='#ff7f0e', lw=2.5, label=f'XGBoost (AP = {xgb_pr_auc:.3f})')
    axes[1].plot(rnn_recall, rnn_precision, color='#2ca02c', lw=2.5, label=f'RNN - LSTM (AP = {rnn_pr_auc:.3f})')
    axes[1].axhline(y=y_test.mean(), color='gray', lw=1.5, linestyle='--', label=f'Baseline (AP = {y_test.mean():.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Precision (PPV)', fontsize=12)
    axes[1].set_title('Precision-Recall (PR) Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plot_curves_path = os.path.join(OUTPUT_DIR, "roc_pr_comparison.png")
    plt.savefig(plot_curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison curves to {plot_curves_path}")
    
    # Plot 2: Confusion Matrices (All 3 Models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xticklabels(['No Sepsis', 'Sepsis'])
    axes[0].set_yticklabels(['No Sepsis', 'Sepsis'])
    axes[0].set_title('Random Forest', fontsize=14, fontweight='bold')
    
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xticklabels(['No Sepsis', 'Sepsis'])
    axes[1].set_yticklabels(['No Sepsis', 'Sepsis'])
    axes[1].set_title('XGBoost', fontsize=14, fontweight='bold')
    
    sns.heatmap(rnn_cm, annot=True, fmt='d', cmap='Greens', ax=axes[2], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[2].set_xlabel('Predicted Label', fontsize=12)
    axes[2].set_ylabel('True Label', fontsize=12)
    axes[2].set_xticklabels(['No Sepsis', 'Sepsis'])
    axes[2].set_yticklabels(['No Sepsis', 'Sepsis'])
    axes[2].set_title('RNN - LSTM', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices_comparison.png")
    plt.savefig(plot_cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrices comparison to {plot_cm_path}")
    
    # Save combined results text file
    rnn_summary_path = os.path.join(OUTPUT_DIR, "rnn_model_results.txt")
    with open(rnn_summary_path, 'w') as f:
        f.write("=== COMPARATIVE SEPSIS MODEL PERFORMANCE EVALUATION ===\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Cohort Size: {len(stay_ids)} stays\n")
        f.write(f"Train Set Size: {len(train_idx)} (Sepsis rate: {y_train.mean()*100:.2f}%)\n")
        f.write(f"Test Set Size: {len(test_idx)} (Sepsis rate: {y_test.mean()*100:.2f}%)\n\n")
        
        f.write("--- RANDOM FOREST ---\n")
        f.write(f"ROC-AUC: {rf_auc:.5f}\n")
        f.write(f"PR-AUC (Average Precision): {rf_pr_auc:.5f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {rf_cm[0,0]}  FP: {rf_cm[0,1]}\n")
        f.write(f"  FN: {rf_cm[1,0]}  TP: {rf_cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, rf_pred))
        f.write("\n\n")
        
        f.write("--- XGBOOST ---\n")
        f.write(f"ROC-AUC: {xgb_auc:.5f}\n")
        f.write(f"PR-AUC (Average Precision): {xgb_pr_auc:.5f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {xgb_cm[0,0]}  FP: {xgb_cm[0,1]}\n")
        f.write(f"  FN: {xgb_cm[1,0]}  TP: {xgb_cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, xgb_pred))
        f.write("\n\n")
        
        f.write("--- RNN - LSTM ---\n")
        f.write(f"ROC-AUC: {rnn_auc:.5f}\n")
        f.write(f"PR-AUC (Average Precision): {rnn_pr_auc:.5f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {rnn_cm[0,0]}  FP: {rnn_cm[0,1]}\n")
        f.write(f"  FN: {rnn_cm[1,0]}  TP: {rnn_cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, rnn_pred))
        
    print(f"\nComparative results saved to {rnn_summary_path}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"RNN Model pipeline completed successfully in {duration/60.0:.2f} minutes!")

if __name__ == "__main__":
    main()
