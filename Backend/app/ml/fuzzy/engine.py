"""
fuzzy suitability engine rsi(momentum) + ppo (trend) + ATR ( voltaility) , quantile - calibrated triangular membership functions , all 27 rules genrated programmatically from a weighted scoring table (see RULE_WEIGHT below) calibration is cached per asset and rebuilt on a fixed interval , not every tick.

"""
import logging
import time
from itertools import product

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

logger = logging.getLogger(__name__)

RSI_TERMS = ["oversold", "neutral", "overbought"]
PPO_TERMS = ["bearish", "neutral", "bullish"]
ATR_TERMS = ["low", "normal", "high"]
OUTPUT_TERMS = ["low", "moderate", "high"]

# reasoning: each state's directional contribution to suitability
RULE_WEIGHTS = {
    "rsi": {"oversold": 1.0, "neutral": 0.0, "overbought": -1.0},
    "ppo": {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0},
    "atr": {"low": 0.5, "normal": 0.0, "high": -1.0},
}

HIGH_THRESHOLD = 1.5
LOW_THRESHOLD = -1.5

CACHE_TTL_SECONDS = 4 * 60 * 60  # rebuild every 4 hours
_engine_cache: dict[int, tuple[float, "ctrl.ControlSystemSimulation"]] = {}


def calibrate_quantile_bounds(series: pd.Series, low_q=0.10, mid_q=0.50, high_q=0.90):
    """Returns (low, mid, high) anchor points from an indicator's own history."""
    clean = series.dropna()
    if len(clean) < 100:
        raise ValueError("Need atleast 100 continuous points to calibrate Fuzzy FIS")
    low, mid, high = clean.quantile([low_q, mid_q, high_q])
    return float(low), float(mid), float(high)


def _score_to_bucket(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score <= LOW_THRESHOLD:
        return "low"
    return "moderate"


def _build_rules(rsi, ppo, atr, suitability) -> list:
    """Generates all 27 rules from RULE_WEIGHTS — no hand-typed combinations."""
    rules = []
    for rsi_term, ppo_term, atr_term in product(RSI_TERMS, PPO_TERMS, ATR_TERMS):
        score = (
            RULE_WEIGHTS["rsi"][rsi_term]
            + RULE_WEIGHTS["ppo"][ppo_term]
            + RULE_WEIGHTS["atr"][atr_term]
        )
        bucket = _score_to_bucket(score)
        rules.append(
            ctrl.Rule(
                rsi[rsi_term] & ppo[ppo_term] & atr[atr_term],
                suitability[bucket],
            )
        )
    return rules


def build_fis(rsi_history: pd.Series, ppo_history: pd.Series, atr_pct_history: pd.Series):
    """Builds one fresh calibrated FIS from historical indicator series."""
    rsi_lo, rsi_mid, rsi_hi = calibrate_quantile_bounds(rsi_history)
    ppo_lo, ppo_mid, ppo_hi = calibrate_quantile_bounds(ppo_history)
    atr_lo, atr_mid, atr_hi = calibrate_quantile_bounds(atr_pct_history)

    x_rsi = np.arange(0, 101, 1)
    x_ppo = np.linspace(ppo_history.min(), ppo_history.max(), 200)
    x_atr = np.linspace(atr_pct_history.min(), atr_pct_history.max(), 200)
    x_score = np.arange(0, 101, 1)

    rsi = ctrl.Antecedent(x_rsi, "rsi")
    ppo = ctrl.Antecedent(x_ppo, "ppo")
    atr = ctrl.Antecedent(x_atr, "atr")
    suitability = ctrl.Consequent(x_score, "suitability")

    rsi["oversold"] = fuzz.trimf(x_rsi, [0, rsi_lo, rsi_mid])
    rsi["neutral"] = fuzz.trimf(x_rsi, [rsi_lo, rsi_mid, rsi_hi])
    rsi["overbought"] = fuzz.trimf(x_rsi, [rsi_mid, rsi_hi, 100])

    ppo["bearish"] = fuzz.trimf(x_ppo, [x_ppo.min(), ppo_lo, ppo_mid])
    ppo["neutral"] = fuzz.trimf(x_ppo, [ppo_lo, ppo_mid, ppo_hi])
    ppo["bullish"] = fuzz.trimf(x_ppo, [ppo_mid, ppo_hi, x_ppo.max()])

    atr["low"] = fuzz.trimf(x_atr, [x_atr.min(), atr_lo, atr_mid])
    atr["normal"] = fuzz.trimf(x_atr, [atr_lo, atr_mid, atr_hi])
    atr["high"] = fuzz.trimf(x_atr, [atr_mid, atr_hi, x_atr.max()])

    suitability["low"] = fuzz.trimf(x_score, [0, 0, 45])
    suitability["moderate"] = fuzz.trimf(x_score, [35, 50, 65])
    suitability["high"] = fuzz.trimf(x_score, [55, 100, 100])

    rules = _build_rules(rsi, ppo, atr, suitability)
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def get_or_build_fis(asset_id: int, rsi_history, ppo_history, atr_history):
    """Returns a cached FIS if fresh, otherwise rebuilds and caches it."""
    now = time.time()
    cached = _engine_cache.get(asset_id)

    if cached is not None:
        built_at, sim = cached
        if now - built_at < CACHE_TTL_SECONDS:
            return sim

    sim = build_fis(rsi_history, ppo_history, atr_history)
    _engine_cache[asset_id] = (now, sim)
    return sim


def compute_suitability(
    asset_id: int,
    rsi_history: pd.Series,
    ppo_history: pd.Series,
    atr_pct_history: pd.Series,
    current_rsi: float,
    current_ppo: float,
    current_atr_pct: float,
) -> dict:
    """Runs the FIS on current values, returns crisp suitability score."""
    sim = get_or_build_fis(asset_id, rsi_history, ppo_history, atr_pct_history)

    rsi_in = float(np.clip(current_rsi, 0, 100))
    ppo_in = float(np.clip(current_ppo, ppo_history.min(), ppo_history.max()))
    atr_in = float(np.clip(current_atr_pct, atr_pct_history.min(), atr_pct_history.max()))

    if not all(np.isfinite(v) for v in (rsi_in, ppo_in, atr_in)):
        raise ValueError(
            f"Non-finite FIS input for asset {asset_id}: "
            f"rsi={current_rsi}, ppo={current_ppo}, atr_pct={current_atr_pct}"
        )

    sim.input["rsi"] = rsi_in
    sim.input["ppo"] = ppo_in
    sim.input["atr"] = atr_in

    logger.debug(f"Asset={asset_id} RSI={rsi_in:.2f} PPO={ppo_in:.2f} ATR_PCT={atr_in:.2f}%")
    sim.compute()

    if "suitability" not in sim.output:
        raise ValueError(
            f"FIS produced no output for asset {asset_id}: "
            f"rsi={rsi_in}, ppo={ppo_in}, atr={atr_in}"
        )
    score = float(sim.output["suitability"])

    return {"suitability_score": round(score, 2)}