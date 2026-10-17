import asyncio
from datetime import datetime, timezone

from app.models.userauth import User, VerificationToken, PersonalProfile
from app.models.portfolio_auth import PortfolioInfo, Assets, PortfolioSources, MarketPriceCandles
from app.models.nse import MarketInstruments, NSEPriceCandles, MarketBreadth
from app.connectors.nse.indices import NiftyConnector
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db.connection import get_engine

SessionLocal = async_sessionmaker(
    bind=get_engine(),
    autoflush=False,
    expire_on_commit=False,
)

async def main():
    async with SessionLocal() as db:
        connector = NiftyConnector(
            db=db,
            instrument_id=1,
            exchange="NSE",
            timeframe="1m",
            yf_symbol="^NSEI",
        )

        start = datetime(2026, 8, 11 , tzinfo=timezone.utc)
        end = datetime(2026, 8, 15, tzinfo=timezone.utc)

        raw_data = await connector.fetch(start, end)
        normalized = connector.normalize(raw_data)
        good, bad = connector.validate(normalized)

        print(f"fetched: {len(normalized)}, good: {len(good)}, bad: {len(bad)}")
        if good:
            print("sample row:", good[0])

asyncio.run(main())