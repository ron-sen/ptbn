
"""
read size queries for the dashboard and market screens  
"""

import os 
from collections import defaultdict 
from datetime import datetime , timedelta , timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import and_ , func  , or_ , select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.indicator import IndicatorSignal , IndicatorValues , SuitabilityScores

from app.models.portfolio_auth import (
    Assets , MarketPriceCandles , PortfolioInfo , PortfolioPosition
)

from app.models.userauth import PersonalProfile


# config 

DASHBOARD = ["BTC" , "ETH" , "SOL" , "DOGE"]

def _env_exchange() -> Optional[str]:
    value = os.getenv("PTBN_CANDLE_EXCHANGE" , "AGGREGATED").strip()
    return None if value in ("" , "*") else value

# exchange value --> connectors write into marketpricecandles 

CANDLE_EXCHANGE = _env_exchange()

# timeframe my indicator pipline compute on  which was 1 min candles 

INDICATOR_TIMEFRAME = os.getenv("PTBN_INDICATOR_TIMEFRAME" , "1m")

# timeftrame used for 24h stats and sparklines

SNAPSHOT_TIMEFRAME = os.getenv("PTBN_SNAPSHOT_TIMEFRAME" , "1h")

PRICE_TF_PREFERENCE = os.getenv("1m", "5m", "15m", "1h", "4h", "1d")

ALLOWED_TIMEFRAMES = set(PRICE_TF_PREFERENCE)
MAX_CANDLES = 1000
MAX_BROWSE = 300 
STABLE_TYPES = {"STABLECOIN" , "FLAT"}
DOMINANCE_DAYS = {"week" : 7  , "month" : 30}
DOMINANCE_TOP_N = 3 

HIGH_MIN , MODERATE_MIN = 75 , 40  # display band for the suitabilty index

SORT_KEYS = { "name" , "symbol" , "price" , "change" , "volume"}

def _f(x) -> float :
    return float(x) if x is not None else 0.0 

def _fo(x) -> Optional[float]:
    return float(x)  if x is not None else None 

def _now() -> datetime :
    return datetime.now(timezone.utc)

def _exch() -> list :
    return [MarketPriceCandles.exchange == CANDLE_EXCHANGE] if CANDLE_EXCHANGE else []

def band(score : float) -> str :
    if score >= HIGH_MIN :
        return "High"
    if score >= "MODERATE_MIN":
        return "Moderate"
    return "Low"

def _ind_row(r : IndicatorValues) -> dict :
    return {
        "time": r.timestamp,
        "sma": _f(r.sma), "ema": _f(r.ema), "rsi": _f(r.rsi),
        "macd_line": _f(r.macd_line), "macd_signal": _f(r.macd_signal), "macd_hist": _f(r.macd_hist),
        "ppo": _fo(r.ppo),
        "bb_upper": _f(r.bb_upper), "bb_middle": _f(r.bb_middle), "bb_lower": _f(r.bb_lower),
        "bb_percent_b": _f(r.bb_percent_b), "bb_bandwidth": _f(r.bb_bandwidth),
        "atr": _f(r.atr), "adx": _f(r.adx), "plus_di": _f(r.plus_di), "minus_di": _f(r.minus_di),
        "obv": _f(r.obv), "vwap": _f(r.vwap), "stoch_k": _f(r.stoch_k), "stoch_d": _f(r.stoch_d),
    }


# sginal log vocab field -> (indicator label , {new_state: event text}) )

FIELD_META = {
    "macd_crossover": ("MACD", {
        "bullish": "Bullish crossover observed", "bearish": "Bearish crossover observed",
        "none": "Crossover state cleared"}),
    "rsi_status": ("RSI", {
        "overbought": "Moved into overbought", "oversold": "Moved into oversold",
        "neutral": "Returned to neutral range"}),
    "bb_status": ("Bollinger Bands", {
        "squeeze": "Volatility squeeze detected", "overbought": "Price pushed above the band",
        "oversold": "Price pushed below the band", "normal": "Returned inside the normal band"}),
    "stoch_crossover": ("Stochastic", {
        "bullish": "Bullish %K/%D crossover", "bearish": "Bearish %K/%D crossover",
        "none": "Crossover state cleared"}),
    "obv_trend": ("OBV", {
        "rising": "Trend turned up", "falling": "Trend turned down", "flat": "Trend flattened"}),
    "vwap_position": ("VWAP", {
        "above": "Price moved above VWAP", "below": "Price moved below VWAP",
        "at": "Price returned to VWAP"}),
    "atr_vol_regime": ("ATR", {
        "high": "Volatility expansion threshold tripped", "low": "Volatility compressed",
        "normal": "Volatility returned to normal"}),
    "adx_trend_strength": ("ADX", {
        "weak": "Trend weakened", "strong": "Trend strengthened", "very_strong": "Very strong trend"}),
}
# Value shown next to an event (column on indicator_values; None = use signal's vwap_deviation_pct)
EVENT_VALUE = {
    "macd_crossover": "macd_hist", "rsi_status": "rsi", "bb_status": "bb_bandwidth",
    "stoch_crossover": "stoch_k", "obv_trend": "obv", "vwap_position": None,
    "atr_vol_regime": "atr", "adx_trend_strength": "adx",
}
TONES = {
    "bullish": "up", "rising": "up", "above": "up", "expanding": "up", "strong": "up", "very_strong": "up",
    "bearish": "down", "falling": "down", "below": "down", "contracting": "down",
    "overbought": "warn", "oversold": "warn", "squeeze": "warn", "high": "warn",
}



class DashboardService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------ helpers
    async def _get_asset(self, asset_id: int) -> Assets:
        asset = (await self.db.execute(select(Assets).where(Assets.asset_id == asset_id))).scalars().first()
        if asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return asset

    async def latest_prices(self, ids: list[int]) -> dict[int, float]:
        """Latest close per asset, preferring the freshest timeframe that has data."""
        prices: dict[int, float] = {}
        remaining = set(ids)
        for tf in PRICE_TF_PREFERENCE:
            if not remaining:
                break
            newest = (
                select(MarketPriceCandles.asset_id.label("aid"), func.max(MarketPriceCandles.candle_time).label("t"))
                .where(MarketPriceCandles.asset_id.in_(remaining), MarketPriceCandles.timeframe == tf, *_exch())
                .group_by(MarketPriceCandles.asset_id)
                .subquery()
            )
            rows = (
                await self.db.execute(
                    select(MarketPriceCandles.asset_id, MarketPriceCandles.close_price)
                    .join(newest, and_(MarketPriceCandles.asset_id == newest.c.aid,
                                       MarketPriceCandles.candle_time == newest.c.t))
                    .where(MarketPriceCandles.timeframe == tf, *_exch())
                )
            ).all()
            for aid, close in rows:
                prices[aid] = float(close)
                remaining.discard(aid)
        return prices

    async def _first_opens(self, ids: list[int], timeframe: str, since: datetime) -> dict[int, float]:
        """Open price of the first candle at/after `since`, per asset."""
        if not ids:
            return {}
        rows = (
            await self.db.execute(
                select(MarketPriceCandles.asset_id, MarketPriceCandles.open_price)
                .where(MarketPriceCandles.asset_id.in_(ids), MarketPriceCandles.timeframe == timeframe,
                       MarketPriceCandles.candle_time >= since, *_exch())
                .order_by(MarketPriceCandles.asset_id, MarketPriceCandles.candle_time)
            )
        ).all()
        out: dict[int, float] = {}
        for aid, open_price in rows:
            out.setdefault(aid, float(open_price))
        return out

    async def snapshots(self, assets: list[Assets]) -> dict[int, dict]:
        """24h stats for many assets in two queries (no per-asset round trips)."""
        if not assets:
            return {}
        ids = [a.asset_id for a in assets]
        prices = await self.latest_prices(ids)
        rows = (
            await self.db.execute(
                select(MarketPriceCandles.asset_id, MarketPriceCandles.open_price, MarketPriceCandles.high_price,
                       MarketPriceCandles.low_price, MarketPriceCandles.close_price,
                       MarketPriceCandles.volume, MarketPriceCandles.quote_volume)
                .where(MarketPriceCandles.asset_id.in_(ids), MarketPriceCandles.timeframe == SNAPSHOT_TIMEFRAME,
                       MarketPriceCandles.candle_time >= _now() - timedelta(hours=24), *_exch())
                .order_by(MarketPriceCandles.asset_id, MarketPriceCandles.candle_time)
            )
        ).all()
        grouped: dict[int, list] = defaultdict(list)
        for r in rows:
            grouped[r.asset_id].append(r)

        out: dict[int, dict] = {}
        for a in assets:
            candles = grouped.get(a.asset_id, [])
            price = 1.0 if a.asset_type in STABLE_TYPES else prices.get(a.asset_id)
            snap = {
                "asset_id": a.asset_id, "symbol": a.symbol, "name": a.name, "category": a.category,
                "asset_type": a.asset_type, "price": price, "change_24h_pct": None,
                "high_24h": None, "low_24h": None, "volume_24h": None, "sparkline": [],
            }
            if candles:
                last_close = _f(candles[-1].close_price)
                first_open = _f(candles[0].open_price)
                price = price if price is not None else last_close
                snap.update(
                    price=price,
                    change_24h_pct=round((price - first_open) / first_open * 100, 2) if first_open else None,
                    high_24h=max(_f(c.high_price) for c in candles),
                    low_24h=min(_f(c.low_price) for c in candles),
                    volume_24h=sum(
                        _f(c.quote_volume) if c.quote_volume is not None else _f(c.volume) * _f(c.close_price)
                        for c in candles),
                    sparkline=[_f(c.close_price) for c in candles],
                )
            out[a.asset_id] = snap
        return out

    # ------------------------------------------------------------------ assets
    async def list_assets(self, *, search: Optional[str], category: Optional[str], sort_by: str,
                          sort_order: str, offset: int, limit: int, symbols: Optional[list[str]] = None) -> dict:
        if sort_by not in SORT_KEYS:
            raise HTTPException(status_code=422, detail=f"sort_by must be one of {sorted(SORT_KEYS)}")
        stmt = select(Assets).where(Assets.is_active.is_(True), Assets.asset_type == "CRYPTO")
        if symbols:
            stmt = stmt.where(Assets.symbol.in_(symbols))
        if category and category.upper() != "ALL":
            stmt = stmt.where(Assets.category == category.upper())
        if search and search.strip():
            term = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{term}%"
            stmt = stmt.where(or_(Assets.symbol.ilike(pattern, escape="\\"), Assets.name.ilike(pattern, escape="\\")))
        assets = (await self.db.execute(stmt.order_by(Assets.asset_id).limit(MAX_BROWSE))).scalars().all()
        snaps = await self.snapshots(list(assets))

        key = {"name": "name", "symbol": "symbol", "price": "price",
               "change": "change_24h_pct", "volume": "volume_24h"}[sort_by]
        items = list(snaps.values())
        known = [i for i in items if i[key] is not None]
        unknown = [i for i in items if i[key] is None]
        known.sort(key=lambda i: i[key].lower() if isinstance(i[key], str) else i[key],
                   reverse=(sort_order == "desc"))
        ordered = known + unknown
        return {"total": len(ordered), "items": ordered[offset: offset + limit]}

    async def asset_summary(self, asset_id: int) -> dict:
        asset = await self._get_asset(asset_id)
        snap = (await self.snapshots([asset]))[asset_id]
        latest = (
            await self.db.execute(
                select(SuitabilityScores.suitability_score).where(SuitabilityScores.asset_id == asset_id)
                .order_by(SuitabilityScores.timestamp.desc()).limit(1)
            )
        ).scalar_one_or_none()
        pair = venue = None
        if asset.binance_symbol and asset.binance_symbol.upper().startswith(asset.symbol.upper()):
            quote = asset.binance_symbol[len(asset.symbol):]
            pair = f"{asset.symbol} / {quote}" if quote else None
            venue = "Binance spot"
        score = _fo(latest)
        return {**snap, "pair": pair, "venue": venue, "suitability_score": score,
                "suitability_label": band(score) if score is not None else None}

    # ------------------------------------------------------------------ chart + indicators
    async def chart(self, asset_id: int, timeframe: str, limit: int) -> dict:
        if timeframe not in ALLOWED_TIMEFRAMES:
            raise HTTPException(status_code=422, detail=f"timeframe must be one of {sorted(ALLOWED_TIMEFRAMES)}")
        limit = max(1, min(limit, MAX_CANDLES))
        await self._get_asset(asset_id)

        rows = (
            await self.db.execute(
                select(MarketPriceCandles)
                .where(MarketPriceCandles.asset_id == asset_id, MarketPriceCandles.timeframe == timeframe, *_exch())
                .order_by(MarketPriceCandles.candle_time.desc()).limit(limit)
            )
        ).scalars().all()
        rows = list(reversed(rows))
        candles = [
            {"time": r.candle_time, "open": _f(r.open_price), "high": _f(r.high_price),
             "low": _f(r.low_price), "close": _f(r.close_price), "volume": _f(r.volume)}
            for r in rows
        ]
        available = timeframe == INDICATOR_TIMEFRAME and bool(rows)
        indicators: list[dict] = []
        if available:
            ind_rows = (
                await self.db.execute(
                    select(IndicatorValues)
                    .where(IndicatorValues.asset_id == asset_id,
                           IndicatorValues.timestamp >= rows[0].candle_time,
                           IndicatorValues.timestamp <= rows[-1].candle_time)
                    .order_by(IndicatorValues.timestamp)
                )
            ).scalars().all()
            indicators = [_ind_row(r) for r in ind_rows]
        return {"timeframe": timeframe, "indicator_timeframe": INDICATOR_TIMEFRAME,
                "overlays_available": available and bool(indicators),
                "candles": candles, "indicators": indicators}

    async def indicator_history(self, asset_id: int, limit: int, offset: int) -> dict:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        await self._get_asset(asset_id)
        total = (
            await self.db.execute(
                select(func.count()).select_from(IndicatorValues).where(IndicatorValues.asset_id == asset_id)
            )
        ).scalar_one()
        rows = (
            await self.db.execute(
                select(IndicatorValues).where(IndicatorValues.asset_id == asset_id)
                .order_by(IndicatorValues.timestamp.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
        return {"total": total, "items": [_ind_row(r) for r in rows]}

    # ------------------------------------------------------------------ signals
    async def _latest_signal(self, asset_id: int) -> Optional[IndicatorSignal]:
        return (
            await self.db.execute(
                select(IndicatorSignal).where(IndicatorSignal.asset_id == asset_id)
                .order_by(IndicatorSignal.timestamp.desc()).limit(1)
            )
        ).scalars().first()

    async def _latest_values(self, asset_id: int) -> Optional[IndicatorValues]:
        return (
            await self.db.execute(
                select(IndicatorValues).where(IndicatorValues.asset_id == asset_id)
                .order_by(IndicatorValues.timestamp.desc()).limit(1)
            )
        ).scalars().first()

    async def _atr_pct(self, asset_id: int, atr: Optional[float]) -> Optional[float]:
        if atr is None:
            return None
        price = (await self.latest_prices([asset_id])).get(asset_id)
        return round(float(atr) / price * 100, 3) if price else None

    async def latest_signals(self, asset_id: int) -> dict:
        await self._get_asset(asset_id)
        sig = await self._latest_signal(asset_id)
        if sig is None:
            raise HTTPException(status_code=404, detail="No signals computed yet")
        values = await self._latest_values(asset_id)
        return {
            "timestamp": sig.timestamp, "rsi_status": sig.rsi_status, "macd_crossover": sig.macd_crossover,
            "macd_hist_direction": sig.macd_hist_direction, "bb_status": sig.bb_status,
            "atr_vol_regime": sig.atr_vol_regime,
            "atr_pct": await self._atr_pct(asset_id, values.atr if values else None),
            "stoch_status": sig.stoch_status, "stoch_crossover": sig.stoch_crossover,
            "adx_trend_strength": sig.adx_trend_strength, "adx_trend_direction": sig.adx_trend_direction,
            "vwap_position": sig.vwap_position, "vwap_deviation_pct": _f(sig.vwap_deviation_pct),
            "obv_trend": sig.obv_trend, "sma_slope": sig.sma_slope, "ema_slope": sig.ema_slope,
        }

    async def signal_log(self, asset_id: int, bars: int, max_events: int = 50) -> list[dict]:
        """State changes between consecutive indicator_signals rows (newest first)."""
        bars = max(2, min(bars, 500))
        await self._get_asset(asset_id)
        sigs = (
            await self.db.execute(
                select(IndicatorSignal).where(IndicatorSignal.asset_id == asset_id)
                .order_by(IndicatorSignal.timestamp.desc()).limit(bars + 1)
            )
        ).scalars().all()
        sigs = list(reversed(sigs))
        if len(sigs) < 2:
            return []
        vals = (
            await self.db.execute(
                select(IndicatorValues).where(IndicatorValues.asset_id == asset_id,
                                              IndicatorValues.timestamp.in_([s.timestamp for s in sigs]))
            )
        ).scalars().all()
        by_ts = {v.timestamp: v for v in vals}

        events: list[dict] = []
        for prev, cur in zip(sigs, sigs[1:]):
            for fld, (label, texts) in FIELD_META.items():
                before, after = getattr(prev, fld), getattr(cur, fld)
                if before == after:
                    continue
                col = EVENT_VALUE[fld]
                if col is None:
                    value = _f(cur.vwap_deviation_pct)
                else:
                    v = by_ts.get(cur.timestamp)
                    value = _fo(getattr(v, col)) if v is not None else None
                events.append({"timestamp": cur.timestamp, "field": fld, "indicator": label,
                               "event": texts.get(after, f"{before} to {after}"),
                               "value": value, "tone": TONES.get(after, "flat")})
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        return events[:max_events]

    # ------------------------------------------------------------------ suitability
    async def suitability_history(self, asset_id: int, limit: int) -> list[dict]:
        limit = max(1, min(limit, 200))
        await self._get_asset(asset_id)
        rows = (
            await self.db.execute(
                select(SuitabilityScores).where(SuitabilityScores.asset_id == asset_id)
                .order_by(SuitabilityScores.timestamp.desc()).limit(limit)
            )
        ).scalars().all()
        return [self._suit_point(r) for r in rows]

    @staticmethod
    def _suit_point(r: SuitabilityScores) -> dict:
        return {"timestamp": r.timestamp, "suitability_score": _f(r.suitability_score),
                "rsi_input": _f(r.rsi_input), "ppo_input": _f(r.ppo_input), "atr_input": _f(r.atr_input)}

    async def suitability_latest(self, asset_id: int) -> dict:
        await self._get_asset(asset_id)
        row = (
            await self.db.execute(
                select(SuitabilityScores).where(SuitabilityScores.asset_id == asset_id)
                .order_by(SuitabilityScores.timestamp.desc()).limit(1)
            )
        ).scalars().first()
        if row is None:
            raise HTTPException(status_code=404, detail="No suitability run yet")
        point = self._suit_point(row)
        narration, footnotes = await self._narrate(asset_id, point["suitability_score"])
        return {**point, "label": band(point["suitability_score"]), "narration": narration, "footnotes": footnotes}

    async def _narrate(self, asset_id: int, score: float) -> tuple[Optional[str], list[str]]:
        """Deterministic narration built only from stored signals and indicator values (no model call)."""
        sig, val = await self._latest_signal(asset_id), await self._latest_values(asset_id)
        if sig is None or val is None:
            return None, []
        atr_pct = await self._atr_pct(asset_id, val.atr)
        rsi_phrase = {"overbought": "overbought", "oversold": "oversold"}.get(sig.rsi_status, "in a neutral range")
        macd_phrase = {"bullish": "a bullish crossover", "bearish": "a bearish crossover"}.get(
            sig.macd_crossover, "no recent crossover")
        atr_text = f"{atr_pct:.2f}% of price" if atr_pct is not None else f"{_f(val.atr):,.2f} in price terms"
        text = (
            f"The composite suitability index is {score:.0f} of 100 ({band(score).lower()}). "
            f"RSI(14) at {_f(val.rsi):.1f} is {rsi_phrase}[1]. "
            f"MACD shows {macd_phrase} with the histogram {sig.macd_hist_direction}[2], and ATR at {atr_text} "
            f"indicates a {sig.atr_vol_regime} volatility regime[3]."
        )
        stamp = f"{sig.timestamp:%Y-%m-%d %H:%M} UTC"
        footnotes = [
            f"RSI {_f(val.rsi):.1f}, {sig.rsi_status}, 14 periods, {stamp}",
            f"MACD histogram {_f(val.macd_hist):+.2f}, {sig.macd_hist_direction}, crossover: {sig.macd_crossover}",
            f"ATR {atr_text}, {sig.atr_vol_regime} volatility regime",
        ]
        return text, footnotes

    # ------------------------------------------------------------------ market-wide
    async def volume_dominance(self, range_: str) -> dict:
        days = DOMINANCE_DAYS.get(range_)
        if days is None:
            raise HTTPException(status_code=422, detail="range must be week or month")
        volume = func.sum(func.coalesce(MarketPriceCandles.quote_volume,
                                        MarketPriceCandles.volume * MarketPriceCandles.close_price))
        rows = (
            await self.db.execute(
                select(Assets.symbol, volume.label("vol"))
                .join(Assets, Assets.asset_id == MarketPriceCandles.asset_id)
                .where(MarketPriceCandles.timeframe == "1d",
                       MarketPriceCandles.candle_time >= _now() - timedelta(days=days),
                       Assets.asset_type == "CRYPTO", *_exch())
                .group_by(Assets.symbol).order_by(volume.desc())
            )
        ).all()
        volumes = [(r.symbol, _f(r.vol)) for r in rows if r.vol is not None]
        total = sum(v for _, v in volumes)
        if total <= 0:
            return {"total": 0.0, "items": []}
        items = volumes[:DOMINANCE_TOP_N]
        other = sum(v for _, v in volumes[DOMINANCE_TOP_N:])
        if other > 0:
            items.append(("Other", other))
        return {"total": total,
                "items": [{"symbol": s, "share_pct": round(v / total * 100, 1), "volume": v} for s, v in items]}

    # ------------------------------------------------------------------ me / portfolio
    async def me(self, current_user) -> dict:
        profile = (
            await self.db.execute(select(PersonalProfile).where(PersonalProfile.user_id == current_user.id))
        ).scalars().first()
        return {"id": current_user.id, "mail": current_user.mail,
                "username": profile.username if profile else None,
                "first_name": profile.first_name if profile else None,
                "last_name": profile.last_name if profile else None}

    async def portfolio_summary(self, current_user) -> dict:
        portfolio = (
            await self.db.execute(
                select(PortfolioInfo)
                .where(PortfolioInfo.user_id == current_user.id, PortfolioInfo.is_archived.is_(False))
                .order_by(PortfolioInfo.is_default.desc(), PortfolioInfo.portfolio_id.asc()).limit(1)
            )
        ).scalars().first()
        empty = {"portfolio_id": None, "total_usd": 0.0, "updated_at": None, "change_24h_usd": None,
                 "change_24h_pct": None, "change_30d_pct": None, "holdings": []}
        if portfolio is None:
            return empty

        rows = (
            await self.db.execute(
                select(PortfolioPosition, Assets)
                .join(Assets, Assets.asset_id == PortfolioPosition.asset_id)
                .where(PortfolioPosition.portfolio_id == portfolio.portfolio_id, PortfolioPosition.quantity > 0)
            )
        ).all()
        if not rows:
            return {**empty, "portfolio_id": portfolio.portfolio_id}

        priced = [a.asset_id for _, a in rows if a.asset_type not in STABLE_TYPES]
        prices = await self.latest_prices(priced)
        open_24h = await self._first_opens(priced, SNAPSHOT_TIMEFRAME, _now() - timedelta(hours=24))
        open_30d = await self._first_opens(priced, "1d", _now() - timedelta(days=30))

        def price_of(a: Assets) -> Optional[float]:
            return 1.0 if a.asset_type in STABLE_TYPES else prices.get(a.asset_id)

        holdings, total = [], 0.0
        effect_24h = base_24h = effect_30d = base_30d = 0.0
        latest_update: Optional[datetime] = None
        for pos, asset in rows:
            qty, price = _f(pos.quantity), price_of(asset)
            value = qty * price if price is not None else 0.0
            total += value
            if price is not None:
                p24 = 1.0 if asset.asset_type in STABLE_TYPES else open_24h.get(asset.asset_id, price)
                p30 = 1.0 if asset.asset_type in STABLE_TYPES else open_30d.get(asset.asset_id, price)
                effect_24h += qty * (price - p24)
                base_24h += qty * p24
                effect_30d += qty * (price - p30)
                base_30d += qty * p30
            if latest_update is None or pos.updated_at > latest_update:
                latest_update = pos.updated_at
            holdings.append({"asset_id": asset.asset_id, "symbol": asset.symbol, "name": asset.name,
                             "quantity": qty, "price": price, "value_usd": value, "weight_pct": 0.0})
        for h in holdings:
            h["weight_pct"] = round(h["value_usd"] / total * 100, 2) if total else 0.0
        holdings.sort(key=lambda h: h["value_usd"], reverse=True)

        return {
            "portfolio_id": portfolio.portfolio_id, "total_usd": total, "updated_at": latest_update,
            "change_24h_usd": round(effect_24h, 2),
            "change_24h_pct": round(effect_24h / base_24h * 100, 2) if base_24h else None,
            "change_30d_pct": round(effect_30d / base_30d * 100, 2) if base_30d else None,
            "holdings": holdings,
        }
