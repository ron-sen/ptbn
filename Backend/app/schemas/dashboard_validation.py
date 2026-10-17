
from datetime import datetime 
from typing import Optional

from pydantic import BaseModel

# asset snapshot

class AssetSnapshot(BaseModel):
    """ asset with its last 24 h market stats , if stats are none when no candles exist """

    asset_id : int 
    symbol : str 
    name : str 
    category : Optional[str] = None 
    asset_type : str 
    price : Optional[float] = None 
    change_24_pct : Optional[float] = None 
    high_24  : Optional[float]  = None 
    low_24 : Optional[float] = None 
    volume_24 : Optional[float] = None 
    sparkline : list[float] = None 


class AssetPage(BaseModel):
    total: int 
    items = list[AssetSnapshot]

class AssetSummary(AssetSnapshot):

    pair : Optional[str] = None 
    venue : Optional[str] = None 
    suitability_score : Optional[float] = None 
    suitability_label : Optional[str] = None 

class DominanceItem(BaseModel):

    symbol : str 
    share_ptc : float 
    volume : float

class DominanceResponse(BaseModel):
    total : float 
    items : list[DominanceItem]

# candles 

class CandleOut(BaseModel):

    time : datetime
    open : float
    high : float
    low : float 
    close : float 
    volume : float 

class IndicatorRow(BaseModel):

    time : datetime
    sma : float 
    ema : float 
    rsi : float 
    macd_line : float 
    macd_hist : float 
    ppo : Optional[float] = None 
    bb_uper : float 
    bb_middle : float 
    bb_lower : float 
    bb_percent_b : float 
    bb_bandwidth : float 
    atr : float
    adx : float 
    plus_di : float 
    minus_di : float 
    obv : float 
    vwap : float 
    stoch_k : float 
    stoch_d : float 


class ChartResponse(BaseModel):
    timeframe : str 
    indicator_timeframe : str 
    overlays_available : bool 
    candles : list[CandleOut]
    indicators : list[IndicatorRow]

class IndicatorPage(BaseModel):
    total : int 
    items : list[IndicatorRow]


# signals 

class SignalsLatest(BaseModel):
    timestamp : datetime
    rsi_status: str 
    macd_crossover : str 
    macd_hist_direction : str 
    bb_status : str 
    atr_vol_regime : str 
    atr_pct : Optional[float] = None 
    stoch_status : str 
    adx_trend_strength: str
    adx_trend_direction: str
    vwap_position: str
    vwap_deviation_pct: float
    obv_trend: str
    sma_slope: str
    ema_slope: str


class SignalEvent(BaseModel):
    timestamp: datetime
    field: str                       
    indicator: str                      
    event: str                          
    value: Optional[float] = None
    tone: str                           


# suitability 

class SuitabilityPoint(BaseModel):
    timestamp: datetime
    suitability_score: float
    rsi_input: float
    ppo_input: float
    atr_input: float


class SuitabilityLatest(SuitabilityPoint):
    label: str                        
    narration: Optional[str] = None
    footnotes: list[str] = []


# portfolios 

class MeResponse(BaseModel):
    id: int
    mail: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class HoldingOut(BaseModel):
    asset_id: int
    symbol: str
    name: str
    quantity: float
    price: Optional[float] = None
    value_usd: float
    weight_pct: float


class PortfolioSummary(BaseModel):
    portfolio_id: Optional[int] = None
    total_usd: float
    updated_at: Optional[datetime] = None
    change_24h_usd: Optional[float] = None
    change_24h_pct: Optional[float] = None
    change_30d_pct: Optional[float] = None  
    holdings: list[HoldingOut]
