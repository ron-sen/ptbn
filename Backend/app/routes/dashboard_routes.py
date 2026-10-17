from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connection import get_db
from app.schemas.dashboard_validation import AssetPage, AssetSnapshot, DominanceResponse, MeResponse
from app.security.jwtsecure import get_current_user
from app.services.dashboard_services import DASHBOARD_SYMBOLS, DashboardService

router = APIRouter(tags=["Dashboard"], dependencies=[Depends(get_current_user)])


@router.get("/me", response_model=MeResponse)
async def me(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    return await DashboardService(db).me(current_user)


@router.get("/assets", response_model=AssetPage)
async def list_assets(
    search: Optional[str] = Query(None, max_length=50),
    category: Optional[str] = Query(None, max_length=12),
    sort_by: Literal["name", "symbol", "price", "change", "volume"] = "volume",
    sort_order: Literal["asc", "desc"] = "desc",
    offset: int = Query(0, ge=0),
    limit: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    return await DashboardService(db).list_assets(
        search=search, category=category, sort_by=sort_by, sort_order=sort_order, offset=offset, limit=limit)


@router.get("/market/overview", response_model=list[AssetSnapshot])
async def market_overview(db: AsyncSession = Depends(get_db)):
    page = await DashboardService(db).list_assets(
        search=None, category=None, sort_by="symbol", sort_order="asc", offset=0, limit=50,
        symbols=DASHBOARD_SYMBOLS)
    return sorted(page["items"], key=lambda i: DASHBOARD_SYMBOLS.index(i["symbol"]))


@router.get("/market/volume-dominance", response_model=DominanceResponse)
async def volume_dominance(range: Literal["week", "month"] = "week", db: AsyncSession = Depends(get_db)):
    return await DashboardService(db).volume_dominance(range)
