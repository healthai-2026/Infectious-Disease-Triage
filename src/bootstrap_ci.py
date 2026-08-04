"""
bootstrap_ci.py
---------------
Compute 95% bootstrap confidence intervals for AUROC and AUPRC
from saved out-of-fold prediction arrays.

No model retraining needed — loads oof_predictions.npz produced by
unified_benchmark.py (--save-oof flag) and resamples with replacement.

Usage:
    python src/bootstrap_ci.py
    python src/bootstrap_ci.py --n-boot 2000 --oof reports/oof_predictions.npz
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import datetime
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------------------------- #
#  CLI                                                                          #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--oof",    default=r"C:\PS1\Infectious-Disease-Triage\reports\oof_predictions.npz",
                    help="Path to oof_predictions.npz saved by unified_benchmark.py")
parser.add_argument("--n-boot", type=int, default=1000,
                    help="Number of bootstrap resamples (1000 = ~10 min on CPU)")
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--results-json", default=r"C:\PS1\Infectious-Disease-Triage\reports\unified_benchmark_results.json",
                    help="Results JSON to update with CI values")
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#  Load OOF arrays                                                              #
# ---------------------------------------------------------------------------- #
file_mtime = os.path.getmtime(args.oof)
timestamp = datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
print(f"Loading OOF predictions from: {args.oof}")
print(f"  File Timestamp: {timestamp}")
data      = np.load(args.oof)
seq_y     = data["seq_y"].astype(np.float32)
N         = len(seq_y)
prevalence = float(seq_y.mean())

model_names = [k for k in data.files if k != "seq_y"]
print(f"  Windows  : {N:,}")
print(f"  Positives: {int(seq_y.sum())}  ({prevalence*100:.2f}%)")
print(f"  Models   : {model_names}")
print(f"  Bootstraps: {args.n_boot:,}  (seed={args.seed})")
print()

# ---------------------------------------------------------------------------- #
#  Bootstrap loop                                                               #
# ---------------------------------------------------------------------------- #
rng = np.random.default_rng(args.seed)
ci_results = {}

for name in model_names:
    y_prob = data[name].astype(np.float32)
    aurocs, auprcs = [], []
    t0 = time.time()

    for b in range(args.n_boot):
        idx       = rng.integers(0, N, size=N)
        y_true_b  = seq_y[idx]
        y_prob_b  = y_prob[idx]
        # Skip degenerate bootstrap samples (no positives or all positives)
        if y_true_b.sum() == 0 or y_true_b.sum() == N:
            continue
        aurocs.append(roc_auc_score(y_true_b, y_prob_b))
        auprcs.append(average_precision_score(y_true_b, y_prob_b))

    aurocs = np.array(aurocs)
    auprcs = np.array(auprcs)
    elapsed = time.time() - t0

    auroc_mean  = float(aurocs.mean())
    auroc_lo    = float(np.percentile(aurocs, 2.5))
    auroc_hi    = float(np.percentile(aurocs, 97.5))
    auprc_mean  = float(auprcs.mean())
    auprc_lo    = float(np.percentile(auprcs, 2.5))
    auprc_hi    = float(np.percentile(auprcs, 97.5))

    ci_results[name] = {
        "auroc_boot_mean": round(auroc_mean, 4),
        "auroc_ci_95_lo":  round(auroc_lo,   4),
        "auroc_ci_95_hi":  round(auroc_hi,   4),
        "auprc_boot_mean": round(auprc_mean, 4),
        "auprc_ci_95_lo":  round(auprc_lo,   4),
        "auprc_ci_95_hi":  round(auprc_hi,   4),
        "n_valid_boots":   len(aurocs),
    }

    print(f"  {name:<22}  AUROC {auroc_mean:.4f}  95% CI [{auroc_lo:.4f}, {auroc_hi:.4f}]"
          f"  |  AUPRC {auprc_mean:.4f}  95% CI [{auprc_lo:.4f}, {auprc_hi:.4f}]"
          f"  ({elapsed:.0f}s)")

# ---------------------------------------------------------------------------- #
#  Print summary table                                                          #
# ---------------------------------------------------------------------------- #
print()
print("=" * 80)
print(f"  {'Model':<22}  {'AUROC':>6}  {'95% CI':^20}  {'AUPRC':>6}  {'95% CI':^20}")
print("-" * 80)
for name, r in ci_results.items():
    auroc_ci_str = f"[{r['auroc_ci_95_lo']:.4f}, {r['auroc_ci_95_hi']:.4f}]"
    auprc_ci_str = f"[{r['auprc_ci_95_lo']:.4f}, {r['auprc_ci_95_hi']:.4f}]"
    print(f"  {name:<22}  {r['auroc_boot_mean']:>6.4f}  {auroc_ci_str:^20}  "
          f"{r['auprc_boot_mean']:>6.4f}  {auprc_ci_str:^20}")
print("=" * 80)
print(f"  N bootstraps   : {args.n_boot:,}  (seed={args.seed})")
print(f"  Window prev.   : {prevalence*100:.2f}%  (AUPRC baseline = {prevalence:.4f})")
print("=" * 80)

# ---------------------------------------------------------------------------- #
#  Update results JSON with CI values                                           #
# ---------------------------------------------------------------------------- #
if os.path.exists(args.results_json):
    with open(args.results_json) as f:
        full = json.load(f)

    updated = 0
    for name, ci in ci_results.items():
        if name in full.get("results", {}):
            full["results"][name].update(ci)
            updated += 1

    # Store bootstrap config in metadata
    full.setdefault("bootstrap", {}).update({
        "n_bootstraps": args.n_boot,
        "seed": args.seed,
        "ci_level": "95%",
        "method": "percentile bootstrap on OOF predictions",
        "oof_file": args.oof,
        "oof_file_timestamp": timestamp,
        "n_windows": N,
        "prevalence": round(prevalence, 6),
        "models_updated": updated
    })

    with open(args.results_json, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nUpdated: {args.results_json}  ({updated} models enriched with CI)")
else:
    print(f"\nWARNING: {args.results_json} not found — CI values not merged into results.")
    # Save standalone CI file instead
    out = args.results_json.replace("unified_benchmark_results.json",
                                    "bootstrap_ci_results.json")
    with open(out, "w") as f:
        json.dump({"bootstrap": {"n": args.n_boot, "seed": args.seed}, "ci": ci_results}, f, indent=2)
    print(f"Saved standalone CI file: {out}")

print("\nDone.")
