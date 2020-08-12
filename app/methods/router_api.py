# API router

from fastapi import APIRouter
from app.methods.statistics.router import router as statistics_router
from app.methods.correlation.router import router as correlation_router

router_api = APIRouter()
router_api.include_router(statistics_router)
router_api.include_router(correlation_router)