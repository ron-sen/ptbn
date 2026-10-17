
import asyncio
from datetime import datetime, timezone

from app.models.userauth import User, VerificationToken, PersonalProfile
from app.models.portfolio_auth import PortfolioInfo, Assets, PortfolioSources, MarketPriceCandles
from app.connectors.binance import BinanceConnector
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db.connection import get_engine

SessionLocal = async_sessionmaker(
    bind=get_engine(),
    autoflush=False,
    expire_on_commit=False,
)

async def main() :
    async with SessionLocal() as db :
        connector = BinanceConnector(
            db=db,
            asset_id = 1 , 
            exchange ="AGGREGATED",
            timeframe = "1m" ,
            binance_symbol = "BTCUSDT",
    )
        
        summary = await connector.run()
        print(summary)
        
    
asyncio.run(main())