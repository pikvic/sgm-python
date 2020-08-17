from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import FileResponse

import app.core.config as config
from app.core.queue import get_queue, get_redis, create_task

from app.core.router import router as router_core
from app.methods.router_api import router_api
from app.methods.legacy.router import router as router_legacy


tags_metadata = [
    {
        "name": "core",
        "description": "Служебные операции для работы с задачами, скачиванием результатов и т.д."
    },
    {
        "name": "methods",
        "description": "Методы анализа данных и информация о них"
    },
    {
        "name": "info",
        "description": "Списки методов и групп методов с информацией о них"
    }

]

app = FastAPI(
    title='Вычислительный узел "Многомерный анализ данных"',
    description="Данный вычислительный узел содержит API для различных процедур многомерного анализа данных",
    version="0.2",
    openapi_tags=tags_metadata,
)

app.include_router(router_core, tags=["core"])

if config.DEBUG:
    from app.methods.test.router import router as router_test
    app.include_router(router_test, prefix= "/test", tags=["test"])

app.include_router(router_api, prefix="/api/v1")
app.include_router(router_legacy, tags=["legacy-deprecated"])

if not config.UPLOAD_DIR.exists():
    config.UPLOAD_DIR.mkdir()
if not config.DOWNLOAD_DIR.exists():
    config.DOWNLOAD_DIR.mkdir()

