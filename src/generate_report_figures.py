"""
generate_report_figures.py  —  v2 (5-fold results)
All figures for the PS-I internship report using the 5-fold GroupKFold output.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

OUT = r"C:\PS1\Infectious-Disease-Triage\reports"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "XGBoost":      "#D97706",
    "LSTM":         "#2563EB",
    "RandomForest": "#16A34A",
    "LogisticReg":  "#DC2626",
    "qSOFA":        "#7C3AED",
}
STYLES = {
    "XGBoost":      "--",
    "LSTM":         "-",
    "RandomForest": "-.",
    "LogisticReg":  ":",
    "qSOFA":        (0, (3, 1, 1, 1)),
}
MARKERS = {
    "XGBoost": "D", "LSTM": "o", "RandomForest": "s",
    "LogisticReg": "^", "qSOFA": "x",
}

MODELS = ["LSTM", "XGBoost", "RandomForest", "LogisticReg", "qSOFA"]

# ── 5-fold OOF metrics ────────────────────────────────────────────────────────
metrics = {
    "LSTM":         {"auroc":0.7664,"auprc":0.0245,"lift":4.1,"f1":0.0668,"sens":0.1079,"spec":0.9872,"prec":0.0484,"tp":825, "fp":16231,"fn":6823,"tn":1255716},
    "XGBoost":      {"auroc":0.7935,"auprc":0.0275,"lift":4.6,"f1":0.0703,"sens":0.1321,"spec":0.9842,"prec":0.0479,"tp":1010,"fp":20065,"fn":6638,"tn":1251882},
    "RandomForest": {"auroc":0.7786,"auprc":0.0231,"lift":3.9,"f1":0.0606,"sens":0.0998,"spec":0.9868,"prec":0.0435,"tp":763, "fp":16783,"fn":6885,"tn":1255164},
    "LogisticReg":  {"auroc":0.7554,"auprc":0.0209,"lift":3.5,"f1":0.0594,"sens":0.1148,"spec":0.9835,"prec":0.0401,"tp":878, "fp":21040,"fn":6770,"tn":1250907},
    "qSOFA":        {"auroc":0.5012,"auprc":0.0060,"lift":1.0,"f1":0.0119,"sens":0.4684,"spec":0.5366,"prec":0.0060,"tp":3582,"fp":589386,"fn":4066,"tn":682561},
}

fold_aurocs = {
    "LSTM":         [0.7828, 0.7610, 0.7634, 0.7693, 0.7647],
    "XGBoost":      [0.7916, 0.7751, 0.8095, 0.7997, 0.7995],
    "RandomForest": [0.7799, 0.7553, 0.7866, 0.7842, 0.7865],
    "LogisticReg":  [0.7464, 0.7398, 0.7660, 0.7710, 0.7526],
    "qSOFA":        [0.5110, 0.4941, 0.4830, 0.5096, 0.5066],
}

fold_means = {k: round(float(np.mean(v)),4) for k,v in fold_aurocs.items()}
fold_stds  = {k: round(float(np.std(v)), 4) for k,v in fold_aurocs.items()}

# ROC operating points: [f1-opt, sens50, sens70, sens80]
roc_pts = {
    "LSTM":         {"fpr":[0,1-0.9872,1-0.845,1-0.692,1-0.582,1],"tpr":[0,0.1079,0.500,0.700,0.800,1]},
    "XGBoost":      {"fpr":[0,1-0.9842,1-0.868,1-0.744,1-0.635,1],"tpr":[0,0.1321,0.500,0.700,0.800,1]},
    "RandomForest": {"fpr":[0,1-0.9868,1-0.853,1-0.720,1-0.610,1],"tpr":[0,0.0998,0.500,0.700,0.800,1]},
    "LogisticReg":  {"fpr":[0,1-0.9835,1-0.826,1-0.681,1-0.566,1],"tpr":[0,0.1148,0.500,0.700,0.800,1]},
    "qSOFA":        {"fpr":[0,1-0.9347,1-0.537,1],                 "tpr":[0,0.0604,0.4684,1]},
}

PREV = 0.0060
pr_pts = {
    "LSTM":         {"rec":[0,0.1079,0.500,0.700,0.800,1],"pre":[1,0.0484,0.0148,0.0091,0.0070,PREV]},
    "XGBoost":      {"rec":[0,0.1321,0.500,0.700,0.800,1],"pre":[1,0.0479,0.0166,0.0103,0.0079,PREV]},
    "RandomForest": {"rec":[0,0.0998,0.500,0.700,0.800,1],"pre":[1,0.0435,0.0139,0.0088,0.0067,PREV]},
    "LogisticReg":  {"rec":[0,0.1148,0.500,0.700,0.800,1],"pre":[1,0.0401,0.0127,0.0081,0.0063,PREV]},
    "qSOFA":        {"rec":[0,0.0604,0.4684,1],           "pre":[PREV,0.0055,PREV,PREV]},
}

# ── FIGURE 1 : ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,7))
for name in MODELS:
    fpr = roc_pts[name]["fpr"]
    tpr = roc_pts[name]["tpr"]
    lbl = name + "  (AUROC=" + str(metrics[name]["auroc"]) + ")"
    ax.plot(fpr, tpr, color=COLORS[name], lw=2.2, linestyle=STYLES[name], label=lbl)
ax.plot([0,1],[0,1],"k--",lw=1.0,label="Random Chance (0.5000)")
ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
ax.set_title("Receiver Operating Characteristic Curves\n"
             "Unified Benchmark — 5-fold GroupKFold | Horizon = 6 h", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(0,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_roc_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 1 - ROC saved")

# ── FIGURE 2 : PR Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,7))
for name in MODELS:
    lbl = name + "  (AUPRC=" + str(metrics[name]["auprc"]) + ", lift=" + str(metrics[name]["lift"]) + "x)"
    ax.plot(pr_pts[name]["rec"], pr_pts[name]["pre"],
            color=COLORS[name], lw=2.2, linestyle=STYLES[name], label=lbl)
ax.axhline(PREV, color="gray", ls="--", lw=1.0, label="Random Baseline  (AUPRC=0.0060)")
ax.set_xlabel("Recall  (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision  (Positive Predictive Value)", fontsize=12)
ax.set_title("Precision-Recall Curves\n"
             "Unified Benchmark — 5-fold GroupKFold | Window Prevalence = 0.60%", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.25)
ax.set_xlim(0,1); ax.set_ylim(0,0.14)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_pr_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 2 - PR saved")

# ── FIGURE 3 : Per-Fold AUROC Grouped Bar Chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(5)
w = 0.14
offsets = np.linspace(-0.30, 0.30, len(MODELS))
for i, name in enumerate(MODELS):
    vals = fold_aurocs[name]
    bars = ax.bar(x + offsets[i], vals, w, color=COLORS[name],
                  alpha=0.88, edgecolor="white", linewidth=0.5, label=name)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                str(round(v,3)), ha="center", va="bottom", fontsize=6, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Fold 1\n(val=255,919)","Fold 2\n(val=255,919)","Fold 3\n(val=255,919)",
                    "Fold 4\n(val=255,919)","Fold 5\n(val=255,919)"], fontsize=8)
ax.set_ylim(0.44, 0.86)
ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("Per-Fold Validation AUROC — All Models, Identical Splits\n"
             "5-fold GroupKFold by subject_id | 1,279,595 windows total", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.18))
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_fold_aurocs.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 3 - Per-fold bar saved")

# ── FIGURE 4 : AUROC & AUPRC Summary Bar ────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5))
xi = np.arange(len(MODELS))

ax = axes[0]
aurocs = [metrics[m]["auroc"] for m in MODELS]
bars = ax.bar(xi, aurocs, 0.5, color=[COLORS[m] for m in MODELS], alpha=0.88, edgecolor="white")
for bar, v in zip(bars, aurocs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xi); ax.set_xticklabels(MODELS, fontsize=9, rotation=12, ha="right")
ax.set_ylim(0.45,0.85); ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("AUROC by Model", fontsize=11, fontweight="bold"); ax.grid(alpha=0.25,axis="y")

ax = axes[1]
auprcs = [metrics[m]["auprc"] for m in MODELS]
bars = ax.bar(xi, auprcs, 0.5, color=[COLORS[m] for m in MODELS], alpha=0.88, edgecolor="white")
ax.axhline(PREV, color="gray", ls="--", lw=1.2, label="Random baseline")
for bar, v in zip(bars, auprcs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xi); ax.set_xticklabels(MODELS, fontsize=9, rotation=12, ha="right")
ax.set_ylim(0,0.036); ax.set_ylabel("AUPRC", fontsize=11)
ax.set_title("AUPRC by Model\n(baseline = 0.0060)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25,axis="y")

fig.suptitle("Overall OOF Performance — 5-fold GroupKFold Unified Benchmark", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_auroc_auprc_bar.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 4 - AUROC/AUPRC bar saved")

# ── FIGURE 5 : Confusion Matrices ────────────────────────────────────────────
cmaps = {"LSTM":"Blues","XGBoost":"Oranges","RandomForest":"Greens","LogisticReg":"Reds","qSOFA":"Purples"}
fig, axes = plt.subplots(1,5,figsize=(22,4))
for ax, name in zip(axes, MODELS):
    m = metrics[name]
    cm = np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[name], ax=ax,
                cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"size":9,"weight":"bold"})
    ax.set_title(name+"\nAUROC="+str(m["auroc"])+"\nth="+str(metrics[name]["f1"]),
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
    ax.set_xticklabels(["Neg","Pos"], fontsize=9)
    ax.set_yticklabels(["Neg","Pos"], fontsize=9, rotation=0)
fig.suptitle("Confusion Matrices at F1-Optimal Threshold — 5-fold GroupKFold Unified Benchmark",
             fontsize=11, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_confusion_matrices.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 5 - Confusion matrices saved")

# ── FIGURE 6 : LSTM Training Curves (all 5 folds) ───────────────────────────
fold_data = [
    # (epochs, loss, auroc, early_stop_epoch, best_auroc)
    ([1,2,3,4,5,6,7], [1.3126,1.2853,1.2820,1.2858,1.2374,1.1411,1.0928],
     [0.7736,0.7828,0.7607,0.7525,0.7470,0.7311,0.7282], 7, 0.7828),
    ([1,2,3,4,5,6,7], [1.3031,1.2638,1.2747,1.2574,1.2160,1.0613,1.0600],
     [0.7538,0.7610,0.7526,0.7486,0.7424,0.7332,0.7362], 7, 0.7610),
    ([1,2,3,4,5,6],   [1.2855,1.2793,1.2593,1.2749,1.1814,1.1470],
     [0.7634,0.7554,0.7524,0.7387,0.7391,0.7303],          6, 0.7634),
    ([1,2,3,4,5,6,7], [1.3030,1.2946,1.3071,1.2871,1.2468,1.1304,1.1072],
     [0.7535,0.7693,0.7594,0.7577,0.7431,0.7345,0.7266],  7, 0.7693),
    ([1,2,3,4,5,6,7], [1.3014,1.3156,1.3133,1.2813,1.2329,1.1281,1.0892],
     [0.7623,0.7647,0.7579,0.7460,0.7335,0.7228,0.7262],  7, 0.7647),
]

fig, axes = plt.subplots(1,5,figsize=(22,4.5),sharey=False)
for i,(ax,(ep,lo,au,es,best)) in enumerate(zip(axes,fold_data)):
    ax2 = ax.twinx()
    l1, = ax.plot(ep, lo, color="#2563EB", lw=2, marker="o", ms=4, label="Train Loss")
    l2, = ax2.plot(ep, au, color="#D97706", lw=2, ls="--", marker="s", ms=4, label="Val AUROC")
    best_ep = ep[int(np.argmax(au))]
    ax2.axvline(best_ep, color="gray", ls=":", lw=1, alpha=0.7)
    ax2.annotate("best\nAUROC\n"+str(best), xy=(best_ep, best),
                 xytext=(best_ep+0.2, best-0.01), fontsize=7, color="#D97706")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", color="#2563EB", fontsize=8)
    ax2.set_ylabel("Val AUROC", color="#D97706", fontsize=8)
    ax.set_title("Fold "+str(i+1)+"\n(early stop ep "+str(es)+")", fontsize=9, fontweight="bold")
    ax.set_xticks(ep); ax.grid(alpha=0.2)
    if i==0:
        lines=[l1,l2]; ax.legend(lines,[l.get_label() for l in lines],fontsize=7,loc="upper right")
fig.suptitle("LSTM Training Dynamics — All 5 Folds\n"
             "AdamW | ReduceLROnPlateau (factor=0.5, patience=2) | BCEWithLogitsLoss | Early-stop patience=5",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_lstm_training_curve.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 6 - LSTM training curves saved")

# ── FIGURE 7 : Fixed Operating Points ────────────────────────────────────────
targets = ["Sensitivity ~ 50%", "Sensitivity ~ 70%", "Sensitivity ~ 80%"]
op_spec = {
    "LSTM":         [0.845, 0.692, 0.582],
    "XGBoost":      [0.868, 0.744, 0.635],
    "RandomForest": [0.853, 0.720, 0.610],
    "LogisticReg":  [0.826, 0.681, 0.566],
    "qSOFA":        [0.537, 0.537, 0.000],
}
fig, ax = plt.subplots(figsize=(10,5))
xi = np.arange(len(targets))
offsets = np.linspace(-0.28, 0.28, len(MODELS))
w = 0.12
for i, name in enumerate(MODELS):
    bars = ax.bar(xi+offsets[i], op_spec[name], w,
                  color=COLORS[name], alpha=0.88, edgecolor="white", label=name)
    for bar, v in zip(bars, op_spec[name]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                str(round(v,3)), ha="center", va="bottom", fontsize=7)
ax.set_xticks(xi); ax.set_xticklabels(targets, fontsize=11)
ax.set_ylim(0,1.0); ax.set_ylabel("Achieved Specificity", fontsize=11)
ax.set_title("Specificity at Fixed Sensitivity Operating Points\n"
             "5-fold Unified Benchmark | Higher specificity = fewer false alarms", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.5,-0.20))
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_operating_points.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 7 - Operating points saved")

# ── FIGURE 8 : Mean AUROC with ±1-std error bars ─────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
xi = np.arange(len(MODELS))
means = [fold_means[m] for m in MODELS]
stds  = [fold_stds[m]  for m in MODELS]
bars = ax.bar(xi, means, 0.5, color=[COLORS[m] for m in MODELS],
              alpha=0.88, edgecolor="white", yerr=stds,
              error_kw={"elinewidth":2,"ecolor":"#374151","capsize":6,"capthick":2})
for bar, v, s in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.003,
            str(v)+" ±"+str(s), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(xi); ax.set_xticklabels(MODELS, fontsize=10)
ax.set_ylim(0.44,0.86); ax.set_ylabel("Mean OOF AUROC (±1 std)", fontsize=11)
ax.set_title("Cross-Validated AUROC with Standard Deviation\n"
             "5-fold GroupKFold by subject_id — All Models", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random chance")
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_mean_auroc_errorbars.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Fig 8 - Mean AUROC error bars saved")

print("\nAll 8 figures generated successfully.")
