# API router

from fastapi import APIRouter
from app.methods.statistics.router import router as statistics_router
from app.methods.correlation.router import router as correlation_router
from app.methods.clustering.router import router as clustering_router
from app.methods.preprocessing.router import router as preprocessing_router
from app.methods.factor.router import router as factor_router
from app.methods.regression.router import router as regression_router
from app.methods.methods import GROUPS, METHODS

router_api = APIRouter()

@router_api.get('/methods', tags=['info'])
def methods():
    return METHODS

@router_api.get('/groups', tags=['info'])
def groups():
    return GROUPS

router_api.include_router(statistics_router, prefix='/methods', tags=['methods'])
router_api.include_router(correlation_router, prefix='/methods', tags=['methods'])
router_api.include_router(clustering_router, prefix='/methods', tags=['methods'])
router_api.include_router(preprocessing_router, prefix='/methods', tags=['methods'])
router_api.include_router(factor_router, prefix='/methods', tags=['methods'])
router_api.include_router(regression_router, prefix='/methods', tags=['methods'])
