import asyncio
import numpy as np

from app.db.connection import get_celery_sessionmaker
from app.ml.ml_models.dataset import build_training_dataset
from app.ml.ml_models.train_lr import train_lr_walk_forward
from app.ml.ml_models.train_rf import train_rf_walk_forward
from app.ml.ml_models.evalute_lr import evaluate_all_folds

ASSET_ID = 1  # my real btc asset id


async def main():
    engine, session_factory = get_celery_sessionmaker()

    async with session_factory() as db:
        df = await build_training_dataset(db, asset_id=ASSET_ID)
        await engine.dispose()

        print("\n================ REAL DATASET ================\n")
        print(df.head(15))
        print("\n Dataset shape: ", df.shape)
        print("\n Label distribution: ")
        print(df["label"].value_counts())

        if len(df) < 50:
            print(f"Only {len(df)} rows available, too few for meaningful walk forward folds.")
            return None

        print("\n================ LR TRAINING ================\n")
        results = train_lr_walk_forward(df)
        metrics = evaluate_all_folds(results)

        for result in results:
            print(f"\n----------- Fold {result['fold']} -----------")
            print("True labels:     ", result["y_true"])
            print("Predicted proba: ", np.round(result["y_pred_proba"], 3))
            print("Coefficients:    ", result["model"].coef_)
            print("Intercept:       ", result["model"].intercept_)

        print("\n================ LR EVALUATION ================\n")
        for metric in metrics:
            print(f"\n----------- Fold {metric['fold']} -----------")
            print("Accuracy:          ", metric["accuracy"])
            print("Baseline accuracy: ", metric["baseline_accuracy"])
            print("Beat baseline:     ", metric["beats_baseline"])
            print("Precision:         ", metric["precision"])
            print("Recall:            ", metric["recall"])
            print("F1:                ", metric["f1"])
            print("Confusion matrix:  ", metric["confusion_matrix"])
            print("ROC-AUC:           ", metric["roc_auc"])
            print("Calibration:       ", metric["calibration"])

        print("\n================ LR DONE ================\n")

        print("\n================ RF TRAINING ================\n")
        rf_results = train_rf_walk_forward(df)

        for result in rf_results:
            print(f"\n----------- RF Fold {result['fold']} -----------")
            print("Feature importances:", result["feature_importances"])

        print("\n================ RF EVALUATION ================\n")
        rf_metrics = evaluate_all_folds(rf_results)

        for m in rf_metrics:
            print(f"\n----------- RF Fold {m['fold']} -----------")
            print("Accuracy:          ", m["accuracy"])
            print("Baseline accuracy: ", m["baseline_accuracy"])
            print("Beat baseline:     ", m["beats_baseline"])
            print("Precision:         ", m["precision"])
            print("Recall:            ", m["recall"])
            print("F1:                ", m["f1"])
            print("Confusion matrix:  ", m["confusion_matrix"])
            print("ROC-AUC:           ", m["roc_auc"])

        print("\n================ DONE ================\n")

        return df, results, rf_results, metrics, rf_metrics


if __name__ == "__main__":
    asyncio.run(main())