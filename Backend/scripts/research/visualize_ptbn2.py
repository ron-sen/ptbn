"""
Improved chart functions for LR coefficients, RF importances, and calibration.
Replace the corresponding functions in visualize_ptbn.py with these.
"""

import asyncio
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
BLUE, ORANGE, GREEN, RED, PURPLE = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]

FEATURE_COLS = ["rsi", "ppo", "atr_pct"]
N_SPLITS = 5
SAVE_DIR = "scripts/charts"


# ══════════════════════════════════════════════════════════════════════════════
# 6-IMPROVED. LR COEFFICIENTS — dot plot with CI bands + direction annotation
# ══════════════════════════════════════════════════════════════════════════════
def plot_lr_coefficients(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    coef_records = []
    for fold_i, (tr, te) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            continue
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(Xtr, ytr)
        for feat, coef in zip(FEATURE_COLS, lr.coef_[0]):
            coef_records.append({
                "Fold": fold_i,
                "Feature": feat,
                "Coefficient": coef
            })

    coef_df = pd.DataFrame(coef_records)
    summary = coef_df.groupby("Feature")["Coefficient"].agg(["mean", "std", "min", "max"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Logistic Regression — Feature Coefficients Analysis\n"
                 "(Standardized features: coefficients directly comparable)",
                 fontsize=14, fontweight="bold")

    # Left: dot plot per fold with mean line
    ax = axes[0]
    feat_positions = {f: i for i, f in enumerate(FEATURE_COLS)}
    fold_colors = sns.color_palette("tab10", N_SPLITS)

    for _, row in coef_df.iterrows():
        y_pos = feat_positions[row["Feature"]]
        ax.scatter(row["Coefficient"], y_pos + (row["Fold"] - 2) * 0.12,
                   color=fold_colors[int(row["Fold"])], s=80, alpha=0.85, zorder=3)

    # mean + std band
    for _, row in summary.iterrows():
        y_pos = feat_positions[row["Feature"]]
        ax.barh(y_pos, row["mean"], height=0.05,
                color="white", alpha=0.9, zorder=4)
        ax.errorbar(row["mean"], y_pos, xerr=row["std"],
                    fmt="D", color="white", ms=10, lw=2.5,
                    capsize=6, capthick=2, zorder=5,
                    label="Mean ± Std" if y_pos == 0 else "")
        ax.annotate(f"μ={row['mean']:.3f}", xy=(row["mean"], y_pos + 0.22),
                    ha="center", fontsize=9, color="white", fontweight="bold")

    ax.axvline(0, color="tomato", ls="--", lw=1.5, alpha=0.8, label="Zero (no effect)")
    ax.set_yticks(list(feat_positions.values()))
    ax.set_yticklabels(["RSI", "PPO (%)", "ATR% of Price"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient Value\n(negative = higher feature → lower P(label=1))",
                  fontweight="bold")
    ax.set_title("Coefficient per Fold + Mean ± Std", fontweight="bold")
    fold_handles = [mpatches.Patch(color=fold_colors[i], label=f"Fold {i}") for i in range(N_SPLITS)]
    fold_handles.append(mpatches.Patch(color="white", label="Mean ± Std"))
    ax.legend(handles=fold_handles, fontsize=8, loc="lower right")

    # Right: violin plot showing full distribution per feature
    ax2 = axes[1]
    feat_display = {"rsi": "RSI", "ppo": "PPO (%)", "atr_pct": "ATR% of Price"}
    coef_df["Feature_Label"] = coef_df["Feature"].map(feat_display)
    feat_colors = {"RSI": BLUE, "PPO (%)": ORANGE, "ATR% of Price": PURPLE}

    sns.violinplot(data=coef_df, x="Feature_Label", y="Coefficient",
                   palette=feat_colors, ax=ax2, inner="point",
                   order=["RSI", "PPO (%)", "ATR% of Price"])
    ax2.axhline(0, color="tomato", ls="--", lw=1.5, alpha=0.8)

    # annotate direction
    for feat_label, color in feat_colors.items():
        mean_val = coef_df[coef_df["Feature_Label"] == feat_label]["Coefficient"].mean()
        direction = "↓ Mean-reversion?" if mean_val < 0 else "↑ Momentum?"
        ax2.annotate(direction,
                     xy=(list(feat_colors.keys()).index(feat_label), mean_val),
                     xytext=(list(feat_colors.keys()).index(feat_label), mean_val + 0.15),
                     ha="center", fontsize=8, color="white",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1))

    ax2.set_title("Coefficient Distribution (Violin)\nAnnotated with directional interpretation",
                  fontweight="bold")
    ax2.set_xlabel("Feature", fontweight="bold")
    ax2.set_ylabel("Coefficient", fontweight="bold")

    plt.tight_layout()
    path = f"{SAVE_DIR}/6_lr_coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7-IMPROVED. RF IMPORTANCES — horizontal bar + radar + stability plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_rf_importances(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    imp_records = []
    for fold_i, (tr, te) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                    class_weight="balanced", random_state=42)
        rf.fit(Xtr, ytr)
        for feat, imp in zip(FEATURE_COLS, rf.feature_importances_):
            imp_records.append({"Fold": fold_i, "Feature": feat, "Importance": imp})

    imp_df = pd.DataFrame(imp_records)
    feat_display = {"rsi": "RSI", "ppo": "PPO (%)", "atr_pct": "ATR% of Price"}
    imp_df["Feature_Label"] = imp_df["Feature"].map(feat_display)
    summary = imp_df.groupby("Feature_Label")["Importance"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=True)

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    fig.suptitle("Random Forest Feature Importances\n"
                 "(How much each feature reduces impurity across all trees)",
                 fontsize=14, fontweight="bold")

    # Left: horizontal bar with error bars
    ax1 = fig.add_subplot(gs[0])
    colors_bar = [BLUE if f == "RSI" else ORANGE if f == "PPO (%)" else PURPLE
                  for f in summary["Feature_Label"]]
    bars = ax1.barh(summary["Feature_Label"], summary["mean"],
                    xerr=summary["std"], color=colors_bar, alpha=0.85,
                    capsize=6, error_kw={"lw": 2, "capthick": 2, "ecolor": "white"})
    ax1.set_xlabel("Mean Importance ± Std", fontweight="bold")
    ax1.set_title("Mean Importance\nAcross All Folds", fontweight="bold")
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax1.text(row["mean"] + row["std"] + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{row['mean']:.3f}", va="center", fontsize=10,
                 color="white", fontweight="bold")
    ax1.set_xlim(0, summary["mean"].max() + summary["std"].max() + 0.06)

    # Middle: importance per fold (line plot showing stability)
    ax2 = fig.add_subplot(gs[1])
    fold_colors = sns.color_palette("tab10", N_SPLITS)
    for fold_i in imp_df["Fold"].unique():
        fold_data = imp_df[imp_df["Fold"] == fold_i].set_index("Feature_Label")["Importance"]
        ax2.plot(["RSI", "PPO (%)", "ATR% of Price"],
                 [fold_data.get(f, 0) for f in ["RSI", "PPO (%)", "ATR% of Price"]],
                 "o-", color=fold_colors[int(fold_i)], lw=2, ms=8, alpha=0.8,
                 label=f"Fold {fold_i}")
    mean_vals = [imp_df[imp_df["Feature_Label"] == f]["Importance"].mean()
                 for f in ["RSI", "PPO (%)", "ATR% of Price"]]
    ax2.plot(["RSI", "PPO (%)", "ATR% of Price"], mean_vals,
             "D--", color="white", lw=2.5, ms=10, label="Mean", zorder=5)
    ax2.set_title("Importance Stability\nAcross Walk-Forward Folds", fontweight="bold")
    ax2.set_ylabel("Importance", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 0.6)

    # Right: stacked bar showing proportional contribution per fold
    ax3 = fig.add_subplot(gs[2])
    pivot = imp_df.pivot_table(index="Fold", columns="Feature_Label", values="Importance")
    pivot = pivot[["RSI", "PPO (%)", "ATR% of Price"]]
    fold_labels = [f"Fold {i}" for i in pivot.index]
    bottom = np.zeros(len(pivot))
    feat_colors_list = [BLUE, ORANGE, PURPLE]
    for feat, color in zip(["RSI", "PPO (%)", "ATR% of Price"], feat_colors_list):
        vals = pivot[feat].values
        ax3.bar(fold_labels, vals, bottom=bottom, color=color, alpha=0.85, label=feat)
        bottom += vals
    ax3.set_title("Proportional Contribution\nPer Fold (stacked)", fontweight="bold")
    ax3.set_ylabel("Importance (sum ≈ 1.0)", fontweight="bold")
    ax3.legend(fontsize=9, loc="upper right")
    ax3.set_ylim(0, 1.1)
    ax3.axhline(1.0, color="white", ls="--", lw=1, alpha=0.5)

    plt.tight_layout()
    path = f"{SAVE_DIR}/7_rf_importances.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10-IMPROVED. CALIBRATION — all folds, reliability diagram, sharpness hist
# ══════════════════════════════════════════════════════════════════════════════
def plot_calibration(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    fig.suptitle("Probability Calibration Analysis — LR & RF\n"
                 "(Reliability diagrams + sharpness histograms across folds)",
                 fontsize=14, fontweight="bold")

    fold_colors = sns.color_palette("tab10", N_SPLITS)
    lr_all_probs, rf_all_probs, all_yte = [], [], []

    for fold_i, (tr, te) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(Xtr, ytr)
        lr_prob = lr.predict_proba(Xte)[:, 1]

        rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                    class_weight="balanced", random_state=42)
        rf.fit(Xtr, ytr)
        rf_prob = rf.predict_proba(Xte)[:, 1]

        lr_all_probs.append(lr_prob)
        rf_all_probs.append(rf_prob)
        all_yte.append(yte)

        # per-fold reliability diagrams
        for ax_idx, (prob, title) in enumerate([(lr_prob, "LR"), (rf_prob, "RF")]):
            ax = fig.add_subplot(gs[0, ax_idx])
            try:
                prob_true, prob_pred = calibration_curve(yte, prob, n_bins=6, strategy="quantile")
                ax.plot(prob_pred, prob_true, "o-",
                        color=fold_colors[fold_i], lw=1.5, ms=6,
                        alpha=0.75, label=f"Fold {fold_i}")
            except Exception:
                pass
            ax.plot([0, 1], [0, 1], "w--", lw=1.2, alpha=0.6)
            ax.set_title(f"{title} — Per-Fold Reliability", fontweight="bold")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.legend(fontsize=7, loc="upper left")

    # Aggregated calibration (all folds pooled)
    ax_agg = fig.add_subplot(gs[0, 2])
    ax_agg.plot([0, 1], [0, 1], "w--", lw=1.5, label="Perfect calibration")

    for all_probs, label, color in [(lr_all_probs, "LR (pooled)", BLUE),
                                     (rf_all_probs, "RF (pooled)", ORANGE)]:
        pooled_prob = np.concatenate(all_probs)
        pooled_yte  = np.concatenate(all_yte)
        try:
            prob_true, prob_pred = calibration_curve(pooled_yte, pooled_prob,
                                                      n_bins=8, strategy="quantile")
            ax_agg.plot(prob_pred, prob_true, "o-", color=color,
                        lw=2.5, ms=9, label=label)
        except Exception:
            pass

    ax_agg.set_title("Aggregated Calibration\n(All folds pooled)", fontweight="bold")
    ax_agg.set_xlabel("Mean Predicted Probability")
    ax_agg.set_ylabel("Fraction of Positives")
    ax_agg.legend(fontsize=9)
    ax_agg.set_xlim([0, 1]); ax_agg.set_ylim([0, 1])

    # Sharpness histograms (bottom row)
    for ax_idx, (all_probs, title, color) in enumerate([
        (lr_all_probs, "LR Predicted Probability Distribution\n(Sharpness)", BLUE),
        (rf_all_probs, "RF Predicted Probability Distribution\n(Sharpness)", ORANGE),
    ]):
        ax = fig.add_subplot(gs[1, ax_idx])
        pooled = np.concatenate(all_probs)
        ax.hist(pooled, bins=40, color=color, alpha=0.7, edgecolor="white", lw=0.5)
        ax.axvline(0.5, color="tomato", ls="--", lw=1.8, label="Decision threshold (0.5)")
        ax.axvline(pooled.mean(), color="white", ls="-", lw=1.8,
                   label=f"Mean prob = {pooled.mean():.3f}")
        ax.set_xlabel("Predicted Probability", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        sharp = np.mean((pooled - 0.5) ** 2)
        ax.text(0.02, 0.92, f"Sharpness = {sharp:.4f}\n(higher = more confident)",
                transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round", fc="black", alpha=0.4))

    # Bottom right: ECE (Expected Calibration Error) bar chart
    ax_ece = fig.add_subplot(gs[1, 2])
    ece_results = []
    for all_probs, label in [(lr_all_probs, "LR"), (rf_all_probs, "RF")]:
        for fold_i, (prob, yte) in enumerate(zip(all_probs, all_yte)):
            try:
                prob_true, prob_pred = calibration_curve(yte, prob, n_bins=6, strategy="quantile")
                ece = np.mean(np.abs(prob_true - prob_pred))
                ece_results.append({"Model": label, "Fold": f"F{fold_i}", "ECE": ece})
            except Exception:
                pass

    ece_df = pd.DataFrame(ece_results)
    sns.barplot(data=ece_df, x="Fold", y="ECE", hue="Model",
                palette={"LR": BLUE, "RF": ORANGE}, ax=ax_ece)
    ax_ece.set_title("Expected Calibration Error (ECE)\nPer Fold — Lower is Better",
                     fontweight="bold")
    ax_ece.set_ylabel("ECE", fontweight="bold")
    ax_ece.set_xlabel("Fold", fontweight="bold")
    ax_ece.legend(fontsize=9)
    ax_ece.axhline(0.05, color="tomato", ls="--", lw=1.5, alpha=0.7, label="Good threshold")

    path = f"{SAVE_DIR}/10_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")



warnings.filterwarnings("ignore")

os.makedirs(SAVE_DIR, exist_ok=True)

async def fetch_data():
    from app.db.connection import get_celery_sessionmaker
    from app.ml.ml_models.dataset import build_training_dataset

    engine, session_factory = get_celery_sessionmaker()
    async with session_factory() as db:
        df = await build_training_dataset(db, asset_id=1, include_label=False)
    await engine.dispose()
    return df



def prepare_df(raw: pd.DataFrame, n_candles: int = 5, k: float = 2.0) -> pd.DataFrame:
    from app.ml.ml_models.labeling import compute_forward_return_label
    df = raw.copy()
    df["label"] = compute_forward_return_label(df, n_candles_ahead=n_candles, atr_multiplier=k)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


async def main():
    print("\n📊 PTBN — Improved Visualization Charts")
    print("=" * 50)
    print("Saving to:", os.path.abspath(SAVE_DIR))
    print("Fetching data from DB...")

    raw_df = await fetch_data()
    print(f"  Base rows: {len(raw_df)}")

    df = prepare_df(raw_df, n_candles=5, k=2.0)
    print(f"  Labeled rows (N=5, k=2.0): {len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"\nGenerating 3 improved charts...\n")

    plot_lr_coefficients(df)
    plot_rf_importances(df)
    plot_calibration(df)

    print("\n✅ Done — charts saved to", os.path.abspath(SAVE_DIR))


if __name__ == "__main__":
    asyncio.run(main())
