# throwaway script, not part of app/ — just paste in a REPL or scratch file
import pandas as pd
import numpy as np
from app.ml.indicators import calculations as calc
from app.ml.indicators import signals as sig

# fake 60 rows of OHLCV so every indicator has enough lookback
idx = pd.date_range("2026-01-01", periods=60, freq="1min", tz="UTC")
df = pd.DataFrame({
    "open": np.random.uniform(99, 101, 60),
    "high": np.random.uniform(101, 103, 60),
    "low": np.random.uniform(97, 99, 60),
    "close": np.random.uniform(99, 101, 60),
    "volume": np.random.uniform(100, 1000, 60),
}, index=idx)

sma = calc.calculate_sma(df["close"])
bb = calc.calculate_bollinger_bands(df["close"])
atr = calc.calculate_atr(df)
obv = calc.calculate_obv(df)
vwap = calc.calculate_vwap(df)

print(sig.derive_sma_slope(sma).tail())
print(sig.derive_bb_status(df["close"], bb).tail())
print(sig.derive_atr_vol_regime(atr).tail())