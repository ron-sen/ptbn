import asyncio
from datetime import datetime, timezone

from app.models.userauth import User, VerificationToken, PersonalProfile
from app.models.portfolio_auth import PortfolioInfo, Assets, PortfolioSources, MarketPriceCandles
from app.connectors.polygon import PolygonConnector
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db.connection import get_engine

SessionLocal = async_sessionmaker(
    bind=get_engine(),
    autoflush=False,
    expire_on_commit=False,
)

# Mock Polygon response — same shape as real API
mock_response = {
    "results": [
        {"t": 1704067200000, "o": 187.15, "h": 189.23, "l": 186.74, "c": 188.63, "v": 42567890.0, "vw": 188.12},
        {"t": 1704153600000, "o": 188.63, "h": 190.10, "l": 187.90, "c": 189.50, "v": 38901234.0, "vw": 189.01},
        {"t": 1704240000000, "o": 189.50, "h": 191.00, "l": 188.20, "c": 190.25, "v": 45123456.0, "vw": 189.80},
    ]
}

async def main():
    async with SessionLocal() as db:
        connector = PolygonConnector(
            db=db,
            asset_id=1,
            exchange="AGGREGATED",
            timeframe="1d",
            polygon_ticker="X:BTCUSD",
        )

        # test normalize
        normalized = connector.normalize(mock_response)
        print(f"\n--- normalize ---")
        print(f"rows produced: {len(normalized)}")
        print(f"sample: {normalized[0]}")

        # test validate
        good, bad = connector.validate(normalized)
        print(f"\n--- validate ---")
        print(f"good: {len(good)}, quarantined: {len(bad)}")

        # test store
        stored = await connector.store(good)
        print(f"\n--- store ---")
        print(f"rows upserted: {stored}")

asyncio.run(main())