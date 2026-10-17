

import asyncio
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Seaborn theme ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
BLUE, ORANGE, GREEN, RED, PURPLE = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]

ASSET_ID    = 1
N_SPLITS    = 5
FEATURE_COLS = ["rsi", "ppo", "atr_pct"]
SAVE_DIR    = "scripts/charts"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── DB fetch ───────────────────────────────────────────────────────────────────
async def fetch_data():
    from app.db.connection import get_celery_sessionmaker
    from app.ml.ml_models.dataset import build_training_dataset

    engine, session_factory = get_celery_sessionmaker()
    async with session_factory() as db:
        df = await build_training_dataset(db, asset_id=ASSET_ID, include_label=False)
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. INDICATOR TIME-SERIES  (RSI / PPO / ATR%)
# ══════════════════════════════════════════════════════════════════════════════
def plot_indicator_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Indicator Time-Series — BTC (1-min candles)", fontsize=15, fontweight="bold")

    # RSI
    axes[0].plot(df["timestamp"], df["rsi"], color=BLUE, lw=0.8, alpha=0.85)
    axes[0].axhline(70, color=RED,   ls="--", lw=1.2, label="Overbought (70)")
    axes[0].axhline(30, color=GREEN, ls="--", lw=1.2, label="Oversold (30)")
    axes[0].fill_between(df["timestamp"], df["rsi"], 70,
                         where=df["rsi"] >= 70, alpha=0.15, color=RED)
    axes[0].fill_between(df["timestamp"], df["rsi"], 30,
                         where=df["rsi"] <= 30, alpha=0.15, color=GREEN)
    axes[0].set_ylabel("RSI", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper right", fontsize=9)

    # PPO
    axes[1].plot(df["timestamp"], df["ppo"], color=ORANGE, lw=0.8, alpha=0.85)
    axes[1].axhline(0, color="white", ls="-", lw=0.8)
    axes[1].fill_between(df["timestamp"], df["ppo"], 0,
                         where=df["ppo"] >= 0, alpha=0.2, color=GREEN, label="Bullish")
    axes[1].fill_between(df["timestamp"], df["ppo"], 0,
                         where=df["ppo"] < 0,  alpha=0.2, color=RED,   label="Bearish")
    axes[1].set_ylabel("PPO (%)", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)

    # ATR%
    axes[2].plot(df["timestamp"], df["atr_pct"], color=PURPLE, lw=0.8, alpha=0.85)
    q_high = df["atr_pct"].quantile(0.90)
    q_low  = df["atr_pct"].quantile(0.10)
    axes[2].axhline(q_high, color=RED,   ls="--", lw=1.2, label=f"High volatility (Q90={q_high:.2f}%)")
    axes[2].axhline(q_low,  color=GREEN, ls="--", lw=1.2, label=f"Low volatility  (Q10={q_low:.2f}%)")
    axes[2].set_ylabel("ATR % of Price", fontweight="bold")
    axes[2].set_xlabel("Timestamp", fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = f"{SAVE_DIR}/1_indicator_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. INDICATOR DISTRIBUTIONS  (histograms + KDE + quantile lines)
# ══════════════════════════════════════════════════════════════════════════════
def plot_indicator_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Indicator Distributions with Quantile Calibration Boundaries",
                 fontsize=14, fontweight="bold")

    specs = [
        ("rsi",     "RSI",          BLUE,   [(0.10, "Q10"), (0.50, "Q50"), (0.90, "Q90")]),
        ("ppo",     "PPO (%)",      ORANGE, [(0.10, "Q10"), (0.50, "Q50"), (0.90, "Q90")]),
        ("atr_pct", "ATR% of Price",PURPLE, [(0.10, "Q10"), (0.50, "Q50"), (0.90, "Q90")]),
    ]

    for ax, (col, label, color, quantiles) in zip(axes, specs):
        sns.histplot(df[col], bins=60, kde=True, color=color, alpha=0.6, ax=ax)
        for q, name in quantiles:
            val = df[col].quantile(q)
            ax.axvline(val, ls="--", lw=1.5,
                       label=f"{name}={val:.2f}",
                       color="white" if name == "Q50" else ("tomato" if name == "Q90" else "lightgreen"))
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{SAVE_DIR}/2_indicator_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE vs LABEL  (box plots — do indicator values differ by label?)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_vs_label(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Indicator Values by Label (0 = No Move, 1 = Price Rose > k×ATR in N candles)",
                 fontsize=13, fontweight="bold")

    palette = {0: sns.color_palette("Set2")[3], 1: sns.color_palette("Set2")[2]}
    label_names = {0: "No", 1: "Yes"}
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["label"].map(label_names)

    for ax, col, title in zip(axes,
                               ["rsi", "ppo", "atr_pct"],
                               ["RSI", "PPO (%)", "ATR% of Price"]):
        sns.boxplot(data=df_plot, x="Outcome", y=col,
                    palette={"No": palette[0], "Yes": palette[1]}, ax=ax,
                    order=["No", "Yes"], width=0.5)
        sns.stripplot(data=df_plot.sample(min(500, len(df_plot)), random_state=42),
                      x="Outcome", y=col,
                      palette={"No": palette[0], "Yes": palette[1]},
                      ax=ax, order=["No", "Yes"], alpha=0.2, size=2, jitter=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Price rose > k×ATR in N candles?")
        ax.set_ylabel(title)

    plt.tight_layout()
    path = f"{SAVE_DIR}/3_feature_vs_label.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[["rsi", "ppo", "atr_pct", "label"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5,
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Pearson Correlation — Features + Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{SAVE_DIR}/4_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD ROC CURVES  (LR + RF, all folds)
# ══════════════════════════════════════════════════════════════════════════════
def train_and_plot_roc(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Walk-Forward ROC Curves (TimeSeriesSplit)", fontsize=14, fontweight="bold")

    lr_results, rf_results = [], []

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
        fpr, tpr, _ = roc_curve(yte, lr_prob)
        lr_results.append((fpr, tpr, auc(fpr, tpr), fold_i))

        rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                    class_weight="balanced", random_state=42)
        rf.fit(Xtr, ytr)
        rf_prob = rf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, rf_prob)
        rf_results.append((fpr, tpr, auc(fpr, tpr), fold_i))

    colors = sns.color_palette("tab10", N_SPLITS)
    for ax, results, title in zip(axes,
                                   [lr_results, rf_results],
                                   ["Logistic Regression", "Random Forest"]):
        aucs = []
        for fpr, tpr, roc_auc, fold_i in results:
            ax.plot(fpr, tpr, color=colors[fold_i], lw=1.5,
                    label=f"Fold {fold_i} (AUC={roc_auc:.3f})", alpha=0.8)
            aucs.append(roc_auc)
        ax.plot([0, 1], [0, 1], "w--", lw=1.2, label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title(f"{title}\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                     fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = f"{SAVE_DIR}/5_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")
    return lr_results, rf_results


# ══════════════════════════════════════════════════════════════════════════════
# 6. LR COEFFICIENTS ACROSS FOLDS
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
            coef_records.append({"Fold": f"Fold {fold_i}", "Feature": feat, "Coefficient": coef})

    coef_df = pd.DataFrame(coef_records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=coef_df, x="Feature", y="Coefficient", hue="Fold",
                palette="tab10", ax=ax)
    ax.axhline(0, color="white", lw=1.2, ls="--")
    ax.set_title("LR Coefficients Across Walk-Forward Folds\n"
                 "(Consistent negative PPO = mean-reversion signal at short horizon)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Coefficient (standardized features)", fontweight="bold")
    ax.legend(title="Fold", fontsize=9)
    plt.tight_layout()
    path = f"{SAVE_DIR}/6_lr_coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RF FEATURE IMPORTANCES ACROSS FOLDS
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
            imp_records.append({"Fold": f"Fold {fold_i}", "Feature": feat, "Importance": imp})

    imp_df = pd.DataFrame(imp_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest Feature Importances", fontsize=14, fontweight="bold")

    sns.barplot(data=imp_df, x="Feature", y="Importance", hue="Fold",
                palette="tab10", ax=axes[0])
    axes[0].set_title("Per-Fold Importances", fontweight="bold")
    axes[0].set_ylabel("Importance", fontweight="bold")
    axes[0].legend(title="Fold", fontsize=9)

    mean_imp = imp_df.groupby("Feature")["Importance"].mean().reset_index()
    std_imp  = imp_df.groupby("Feature")["Importance"].std().reset_index()
    mean_imp = mean_imp.sort_values("Importance", ascending=False)
    colors_bar = [BLUE, ORANGE, PURPLE]
    axes[1].bar(mean_imp["Feature"], mean_imp["Importance"],
                yerr=std_imp.set_index("Feature").loc[mean_imp["Feature"], "Importance"].values,
                color=colors_bar, alpha=0.8, capsize=6, edgecolor="white")
    axes[1].set_title("Mean Importance ± Std", fontweight="bold")
    axes[1].set_ylabel("Importance", fontweight="bold")

    plt.tight_layout()
    path = f"{SAVE_DIR}/7_rf_importances.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONFUSION MATRICES  (last fold, LR + RF)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(X))
    tr, te = splits[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices — Fold {N_SPLITS-1} (most recent / hardest)",
                 fontsize=13, fontweight="bold")

    for ax, (ModelClass, kwargs, title) in zip(axes, [
        (LogisticRegression,    {"max_iter": 1000, "class_weight": "balanced"}, "Logistic Regression"),
        (RandomForestClassifier, {"n_estimators": 200, "max_depth": 5, "class_weight": "balanced", "random_state": 42}, "Random Forest"),
    ]):
        model = ModelClass(**kwargs)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        cm = confusion_matrix(yte, ypred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    linewidths=1, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        ax.set_xlabel("Predicted", fontweight="bold")

    plt.tight_layout()
    path = f"{SAVE_DIR}/8_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. LABEL ABLATION HEATMAP  (N × k grid of mean ROC-AUC)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation_heatmap():
    # paste your real results from label_ablation.py output here
    data = {
        (5,  0.5): 0.5261, (5,  1.0): 0.5406, (5,  1.5): 0.5521, (5,  2.0): 0.5643,
        (10, 0.5): 0.5406, (10, 1.0): 0.5475, (10, 1.5): 0.5527, (10, 2.0): 0.5612,
        (15, 0.5): 0.5472, (15, 1.0): 0.5450, (15, 1.5): 0.5524, (15, 2.0): 0.5570,
        (20, 0.5): 0.5449, (20, 1.0): 0.5405, (20, 1.5): 0.5423, (20, 2.0): 0.5457,
        (30, 0.5): 0.5451, (30, 1.0): 0.5407, (30, 1.5): 0.5245, (30, 2.0): 0.5374,
    }
    ns = sorted(set(k[0] for k in data))
    ks = sorted(set(k[1] for k in data))
    matrix = np.array([[data[(n, k)] for k in ks] for n in ns])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"k={k}" for k in ks],
                yticklabels=[f"N={n}" for n in ns],
                ax=ax, linewidths=0.5,
                annot_kws={"size": 11, "weight": "bold"},
                vmin=0.52, vmax=0.57)
    ax.set_title("Label Ablation — Mean ROC-AUC (LR, Walk-Forward)\n"
                 "Best: N=5, k=2.0 → 0.5643  |  All values in tight 0.52–0.57 band",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("ATR Multiplier (k)", fontweight="bold")
    ax.set_ylabel("Forward Window (N candles)", fontweight="bold")
    plt.tight_layout()
    path = f"{SAVE_DIR}/9_ablation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CALIBRATION CURVES  (last fold, LR + RF)
# ══════════════════════════════════════════════════════════════════════════════
def plot_calibration(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(X))
    tr, te = splits[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "w--", lw=1.5, label="Perfectly calibrated")

    for ModelClass, kwargs, label, color in [
        (LogisticRegression,    {"max_iter": 1000, "class_weight": "balanced"}, "Logistic Regression", BLUE),
        (RandomForestClassifier, {"n_estimators": 200, "max_depth": 5, "class_weight": "balanced", "random_state": 42}, "Random Forest", ORANGE),
    ]:
        model = ModelClass(**kwargs)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        try:
            prob_true, prob_pred = calibration_curve(yte, prob, n_bins=8, strategy="quantile")
            ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, ms=7, label=label)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives (actual)", fontweight="bold")
    ax.set_title("Probability Calibration — Last Fold\n"
                 "(Closer to diagonal = better calibrated)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    path = f"{SAVE_DIR}/10_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    print("\n📊 PTBN Visualization Suite")
    print("=" * 50)
    print("Fetching data from DB...")
    raw_df = await fetch_data()
    print(f"  Base rows: {len(raw_df)}")

    df = prepare_df(raw_df, n_candles=5, k=2.0)
    print(f"  Labeled rows (N=5, k=2.0): {len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"\nGenerating charts → {SAVE_DIR}/\n")

    plot_indicator_timeseries(df)
    plot_indicator_distributions(df)
    plot_feature_vs_label(df)
    plot_correlation(df)
    train_and_plot_roc(df)
    plot_lr_coefficients(df)
    plot_rf_importances(df)
    plot_confusion_matrices(df)
    plot_ablation_heatmap()
    plot_calibration(df)

    print("\n✅ All 10 charts saved to", SAVE_DIR)
    print("\nChart index:")
    charts = [
        "1_indicator_timeseries    — RSI/PPO/ATR% over time with threshold zones",
        "2_indicator_distributions  — Histograms + KDE + quantile calibration boundaries",
        "3_feature_vs_label         — Box plots: do indicators differ between label 0 and 1?",
        "4_correlation_heatmap      — Pearson correlation matrix (features + label)",
        "5_roc_curves               — Walk-forward ROC curves, all folds, LR + RF",
        "6_lr_coefficients          — LR coefficients across folds (consistent negative PPO)",
        "7_rf_importances           — RF feature importances across folds (ATR% dominance)",
        "8_confusion_matrices       — TP/TN/FP/FN, last fold, LR + RF",
        "9_ablation_heatmap         — 20-combination N×k label ablation ROC-AUC grid",
        "10_calibration             — Probability calibration curves, last fold",
    ]
    for c in charts:
        print(f"  {c}")


if __name__ == "__main__":
    asyncio.run(main())
