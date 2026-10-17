# scripts/gap_check.py
import asyncio
from app.db.connection import get_celery_sessionmaker
from app.ml.ml_models.dataset import build_training_dataset

ASSET_ID = 1

async def main():
    engine, session_factory = get_celery_sessionmaker()
    async with session_factory() as db:
        df = await build_training_dataset(db, asset_id=ASSET_ID)
        await engine.dispose()

    print("Total rows:", len(df))
    print("\nRows immediately before/after the timestamp jump:")
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    df_sorted["gap_minutes"] = df_sorted["timestamp"].diff().dt.total_seconds() / 60
    big_gaps = df_sorted[df_sorted["gap_minutes"] > 5]
    print(big_gaps[["timestamp", "gap_minutes"]])

if __name__ == "__main__":
    asyncio.run(main())