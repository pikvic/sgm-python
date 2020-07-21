from typing import Optional
from starlette import status
from fastapi import APIRouter
from app.core.schema import TaskPostResult
from app.core.queue import create_task
from .schema import TestTaskParams
from .tasks import run_test

router = APIRouter()

@router.post(
    "/test",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Тестировые методы",
    description="Тестовые методы для имитации работы системы"
)
def test(params: TestTaskParams):
    res = create_task(run_test, params.dict())
    return TaskPostResult(**res)