from fastapi import APIRouter
from app.methods.statistics.router import router as statistics_router

router = APIRouter()

router.include_router(statistics_router, prefix='/statistics', tags=['Statistics'])