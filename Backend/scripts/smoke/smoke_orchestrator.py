

import asyncio
from datetime import datetime, timezone

from app.models.userauth import User, VerificationToken, PersonalProfile
from app.models.portfolio_auth import PortfolioInfo, Assets, PortfolioSources, MarketPriceCandles
from app.connectors.orchestrator import MarketOrchestrator
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db.connection import get_engine

SessionLocal = async_sessionmaker(
    bind=get_engine(),
    autoflush=False,
    expire_on_commit=False,
)


async  def main():
    async with SessionLocal() as db :
        orchestrator = MarketOrchestrator(
            db = db ,
            timeframe= "1m"
        )

        summaries = await orchestrator.run()
        print(summaries)

asyncio.run(main())