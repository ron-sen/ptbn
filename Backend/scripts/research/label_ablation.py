"""
Label Ablation Study — systematic comparison of (n_candles_ahead, atr_multiplier)
combinations using the existing compute_forward_return_label function.
Answers: which label definition gives LR the best ROC-AUC signal?

"""

import asyncio
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

from app.db.connection import get_celery_sessionmaker
from app.ml.ml_models.dataset import build_training_dataset
from app.ml.ml_models.labeling import compute_forward_return_label
from app.ml.ml_models.train_lr import FEATURE_COLUMNS, N_SPLITS
from app.ml.ml_models.evalute_lr import evaluate_fold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ASSET_ID = 1

# --- combinations to test ---
N_CANDLES_OPTIONS = [5, 10, 15, 20, 30]
ATR_MULT_OPTIONS = [0.5, 1.0, 1.5, 2.0]


async def run_ablation():
    engine, session_factory = get_celery_sessionmaker()

    async with session_factory() as db:
        # fetch the base dataset once — no label yet
        base_df = await build_training_dataset(db, asset_id=ASSET_ID, include_label=False)
        await engine.dispose()

    if base_df.empty or len(base_df) < 100:
        print(f"Not enough data ({len(base_df)} rows). Aborting.")
        return

    print(f"\nBase dataset: {len(base_df)} rows, columns: {list(base_df.columns)}")
    print(f"\nTesting {len(N_CANDLES_OPTIONS) * len(ATR_MULT_OPTIONS)} combinations...\n")

    results = []

    for n_candles, atr_mult in product(N_CANDLES_OPTIONS, ATR_MULT_OPTIONS):

        # recompute label for this (N, k) combination
        df = base_df.copy()
        df["label"] = compute_forward_return_label(
            df=df,
            n_candles_ahead=n_candles,
            atr_multiplier=atr_mult,
        )
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        if len(df) < 50:
            print(f"N={n_candles:2d}, k={atr_mult:.1f} | too few rows ({len(df)}) — skipping")
            continue

        label_dist = df["label"].value_counts(normalize=True)
        pos_pct = label_dist.get(1, 0) * 100

        X = df[FEATURE_COLUMNS].values
        y = df["label"].values

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_aucs = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            )
            model.fit(X_train_scaled, y_train)

            proba = model.predict_proba(X_test_scaled)[:, 1]
            metrics = evaluate_fold(
                y_true=y_test,
                y_pred_proba=proba,
                train_majority_label=int(np.round(y_train.mean())),
            )

            if metrics["roc_auc"] is not None:
                fold_aucs.append(metrics["roc_auc"])

        if not fold_aucs:
            print(f"N={n_candles:2d}, k={atr_mult:.1f} | no valid folds")
            continue

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)

        results.append({
            "n_candles": n_candles,
            "atr_mult": atr_mult,
            "rows": len(df),
            "pos_pct": round(pos_pct, 1),
            "mean_roc_auc": round(mean_auc, 4),
            "std_roc_auc": round(std_auc, 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
        })

        print(
            f"N={n_candles:2d}, k={atr_mult:.1f} | "
            f"rows={len(df):5d} | pos%={pos_pct:5.1f}% | "
            f"ROC-AUC={mean_auc:.4f} ± {std_auc:.4f}"
        )

    if not results:
        print("No valid results produced.")
        return

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY — sorted by mean ROC-AUC (best first)")
    print("=" * 70)

    results_df = pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False)
    print(results_df[["n_candles", "atr_mult", "rows", "pos_pct",
                       "mean_roc_auc", "std_roc_auc"]].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest combination: N={int(best['n_candles'])} candles, "
          f"k={best['atr_mult']} ATR multiplier "
          f"→ mean ROC-AUC = {best['mean_roc_auc']}")
    print("\nFold-by-fold AUCs for best combination:", best["fold_aucs"])


if __name__ == "__main__":
    asyncio.run(run_ablation())