import asyncio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.db.connection import get_celery_sessionmaker
from app.ml.ml_models.dataset import (
    build_training_dataset, split_dev_holdout, N_CANDLES_AHEAD,
)
from app.ml.ml_models.train_lr import train_lr_walk_forward, FEATURE_COLUMNS
from app.ml.ml_models.train_rf import train_rf_walk_forward
from app.ml.ml_models.significance import (
    block_bootstrap_auc_ci, circular_shift_pvalue, best_direction_auc,
)


ASSET_ID = 1 
N = N_CANDLES_AHEAD

def report(name , y , scores):
    auc , lo , hi = block_bootstrap_auc_ci(y, scores , block = N)
    p = circular_shift_pvalue(y , scores)

    print(f"  {name:<24} AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  p={p:.3f}")


def pooled(results):
    y = np.concatenate([r["y_true"] for r in results])
    s = np.concatenate([r["y_pred_proba"] for r in results])
    idx = np.concatenate([np.asarray(r["test_index"])for r in results])
    return y , s , idx

def holdout_scores(model , scales , dev , holdout):
    X_tr, y_tr = dev[FEATURE_COLUMNS].values, dev["label"].values
    X_te, y_te = holdout[FEATURE_COLUMNS].values, holdout["label"].values
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    model.fit(X_tr, y_tr)
    return y_te, model.predict_proba(X_te)[:, 1]


async def main():
    engine, session_factory = get_celery_sessionmaker()
    async with session_factory() as db:
        df = await build_training_dataset(db, asset_id=ASSET_ID)
    await engine.dispose()

    dev, holdout = split_dev_holdout(df, N)
    print(f"dev rows={len(dev)}  holdout rows={len(holdout)}  "
          f"pos% dev={dev['label'].mean()*100:.1f} holdout={holdout['label'].mean()*100:.1f}")

    # ---- purged walk-forward on dev ----
    lr_res = train_lr_walk_forward(dev, gap=N)
    rf_res = train_rf_walk_forward(dev, gap=N)

    print("\nWALK-FORWARD (pooled out-of-fold, dev only)")
    y, s, idx = pooled(lr_res); report("LR", y, s)
    y, s, idx = pooled(rf_res); report("RF", y, s)

    print("\nSINGLE-FEATURE BASELINES (best direction, same rows)")
    for f in FEATURE_COLUMNS:
        x = dev.loc[idx, f].values
        print(f"  {f:<24} AUC={best_direction_auc(x, y):.4f}")

    # ---- one-shot holdout ----
    print("\nHOLDOUT (touched once)")
    y_h, s_lr = holdout_scores(
        LogisticRegression(max_iter=1000, class_weight="balanced"), True, dev, holdout)
    report("LR", y_h, s_lr)
    _, s_rf = holdout_scores(
        RandomForestClassifier(n_estimators=200, max_depth=5,
                               class_weight="balanced", random_state=42),
        False, dev, holdout)
    report("RF", y_h, s_rf)
    for f in FEATURE_COLUMNS:
        print(f"  {f:<24} AUC={best_direction_auc(holdout[f].values, y_h):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
