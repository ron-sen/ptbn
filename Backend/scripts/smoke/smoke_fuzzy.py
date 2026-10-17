# scripts/fuzzytest.py
import numpy as np
import pandas as pd
from app.ml.fuzzy.engine import compute_suitability

np.random.seed(42)

# fake 180 candles of indicator history — enough for calibrate_quantile_bounds (needs >= 30)
rsi_history = pd.Series(np.random.normal(55, 12, 180)).clip(0, 100)
ppo_history = pd.Series(np.random.normal(0, 1.5, 180))   # PPO as % of price, small range
atr_history = pd.Series(np.random.uniform(0.5, 3.0, 180))  # ATR in price units

result = compute_suitability(
    asset_id=1,
    rsi_history=rsi_history,
    ppo_history=ppo_history,
    atr_history=atr_history,
    current_rsi=28.5,      # oversold-ish
    current_ppo=1.2,        # bullish-ish
    current_atr=0.8,         # low-ish
)

print("First call (builds + caches):", result)

# second call, same asset_id — should reuse the cached FIS, not rebuild
result2 = compute_suitability(
    asset_id=1,
    rsi_history=rsi_history,
    ppo_history=ppo_history,
    atr_history=atr_history,
    current_rsi=72.0,       # overbought-ish
    current_ppo=-1.0,        # bearish-ish
    current_atr=2.5,          # high-ish
)

print("Second call (should be cached, fast):", result2)