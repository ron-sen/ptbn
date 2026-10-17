"""Write path for "Run Suitability Analysis".

The fuzzy engine itself lives in YOUR code. Expose it as:

    # app/fuzzy/engine.py
    def compute_suitability(rsi: float, ppo: float, atr_pct: float) -> dict:
        return {"score": 0..100, "rsi_input": float, "ppo_input": float, "atr_input": float}

If the module is missing the endpoint answers 501 instead of inventing a score.
"""
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models.indicator import SuitabilityScores
from app.services.dashboard_services import DashboardService


def _load_engine():
    try:
        from app.ml.fuzzy.engine import compute_suitability  # noqa: WPS433 (deliberate lazy import)
    except ImportError:
        return None
    return compute_suitability


async def run_suitability(db: AsyncSession, asset_id: int, engine=None) -> dict:
    engine = engine or _load_engine()
    if engine is None:
        raise HTTPException(status_code=501, detail="Suitability engine is not installed (app/fuzzy/engine.py)")

    service = DashboardService(db)
    await service._get_asset(asset_id)

    values = await service._latest_values(asset_id)
    if values is None or values.ppo is None:
        raise HTTPException(status_code=409, detail="Indicators are not ready for this asset yet")
    atr_pct = await service._atr_pct(asset_id, values.atr)
    if atr_pct is None:
        raise HTTPException(status_code=409, detail="No recent price to normalise ATR")

    result = await run_in_threadpool(engine, float(values.rsi), float(values.ppo), float(atr_pct))
    score = float(result["score"])
    if not 0 <= score <= 100:
        raise HTTPException(status_code=500, detail="Engine returned a score outside 0-100")

    db.add(SuitabilityScores(
        asset_id=asset_id, timestamp=datetime.now(timezone.utc), suitability_score=score,
        rsi_input=float(result["rsi_input"]), ppo_input=float(result["ppo_input"]),
        atr_input=float(result["atr_input"]),
    ))
    await db.commit()
    return await service.suitability_latest(asset_id)
