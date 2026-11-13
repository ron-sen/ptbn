
"""
Builds the historical training dataset for LR/RF mdoels . Joins indicator values with candles close prices , computes normalized features and the target label.

"""

import pandas as pd 
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.ml_models.labeling import compute_forward_return_label
from app.models.indicator import IndicatorValues
from app.models.portfolio_auth import MarketPriceCandles

N_CANDLES_AHEAD = 15  # increased after improvision
ATR_MULTIPLIER = 1.0
HOLDOUT_FRAC = 0.2

async def build_training_dataset(
        db : AsyncSession , 
        asset_id : int ,
        exchange : str = "AGGREGATED",
        timeframe : str = "1m",
        include_label: bool = True, 
) -> pd.DataFrame :

    """
    return a DataFrame with columns: timestamp , rsi , ppo, atr ,  label - sorted chronologically , rows without a full forward window dropped

    """

    indicator_stmt = (
        select(IndicatorValues).where(IndicatorValues.asset_id == asset_id).order_by(IndicatorValues.timestamp.asc())
    )

    indicator_rows = (await db.execute(indicator_stmt)).scalars().all()

    candle_stmt = (
        select(MarketPriceCandles).where(MarketPriceCandles.asset_id == asset_id , MarketPriceCandles.exchange == exchange , MarketPriceCandles.timeframe == timeframe,).order_by(MarketPriceCandles.candle_time.asc())
    )
    candle_rows = (await db.execute(candle_stmt)).scalars().all()

    if not indicator_rows or not candle_rows :
        return pd.DataFrame(columns=["timestamp" , "rsi" , "ppo" ,"atr_pct" , "label"])

    indicator_df = pd.DataFrame([
        {
            "timestamp" : r.timestamp,
            "rsi" : float(r.rsi) if r.rsi is not None else None ,
            "ppo" :float(r.ppo) if r.ppo is not None else None ,
            "atr" : float(r.atr) if r.atr is not None else None ,
        }
        for r in indicator_rows
    ])

    candle_df = pd.DataFrame([
        {"timestamp" : c.candle_time , "close" : float(c.close_price)}
        for c in candle_rows
    ])

    merged = pd.merge(indicator_df , candle_df , on = "timestamp" , how = "inner")
    merged = merged.sort_values("timestamp").reset_index(drop = True)
    merged = merged.dropna(subset = ["rsi" , "ppo" , "atr" , "close"])

    # feature engineered  after improvision : computed ATR as a percentage of close price  for scale invariance 

    merged["atr_pct"] = (merged["atr"] / merged["close"]) * 100.0

    if include_label:
        merged["label"] = compute_forward_return_label(
            df=merged,
            n_candles_ahead=N_CANDLES_AHEAD,
            atr_multiplier=ATR_MULTIPLIER,
        )
        merged = merged.dropna(subset=["label"])
        merged["label"] = merged["label"].astype(int)
        return merged[["timestamp", "rsi", "ppo", "atr_pct", "label"]]
    else:
        # return full columns so ablation can recompute label with different N/k
        return merged[["timestamp", "rsi", "ppo", "atr", "atr_pct", "close"]]



# i need to study this function 
async def split_dev_holdout(df : pd.DataFrame , n_candles_ahead : int = N_CANDLES_AHEAD, holdout_frac :float = HOLDOUT_FRAC):
    """
    chronological split of an already labelled df.
    dev = everything before the cut , minus the last N rows(their labels look  into the holdout)
    holdout = the final slice , touched once at the very end
    """

    cut = int(len(df) * ( 1 - holdout_frac))
    dev = df.iloc[: cut - n_candles_ahead].copy()
    holdout = df.iloc[cut:].copy()
    return dev , holdout


