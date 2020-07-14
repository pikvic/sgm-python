from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import FileResponse

import app.core.config as config
from app.core.queue import get_queue, get_redis, create_task

from app.core.router import router as router_core
from app.methods.statistics.router import router as router_statistics
from app.methods.clustering.router import router as router_clustering
from app.methods.factor.router import router as router_factor
from app.methods.regression.router import router as router_regression
#from app.methods.classification.router import router as router_classification
#from app.methods.visual.router import router as router_visual

tags_metadata = [
    {
        "name": "core",
        "description": "Служебные операции для работы с задачами, скачиванием результатов и т.д."
    },
    {
        "name": "statistics",
        "description": "Статистический анализ"
    },
  
    {
        "name": "clustering",
        "description": "Кластерный анализ"
    },
    {
        "name": "factor",
        "description": "Факторный анализ"
    },
    {
        "name": "regression",
        "description": "Регрессионный анализ"
    },
    # {
    #     "name": "visual",
    #     "description": "Визуальный анализ"
    # },
    #  {
    #     "name": "classification",
    #     "description": "Классификационный анализ"
    # },
]


app = FastAPI(
    title='Вычислительный узел "Многомерный анализ данных"',
    description="Данный вычислительный узел содержит API для различных процедур многомерного анализа данных",
    version="0.1",
    openapi_tags=tags_metadata,
)

app.include_router(router_core, tags=["core"])
app.include_router(router_statistics, prefix="/statistics", tags=["statistics"])
app.include_router(router_clustering, prefix="/clustering", tags=["clustering"])
app.include_router(router_factor, prefix="/factor", tags=["factor"])
app.include_router(router_regression, prefix="/regression", tags=["regression"])
#app.include_router(router_clustering, prefix="/classification", tags=["classification"])
#app.include_router(router_visual, prefix="/visual", tags=["visual"])


if not config.UPLOAD_DIR.exists():
    config.UPLOAD_DIR.mkdir()
if not config.DOWNLOAD_DIR.exists():
    config.DOWNLOAD_DIR.mkdir()

