from typing import Optional
from starlette import status
from fastapi import APIRouter

from app.core.schema import TaskPostResult
from app.core.queue import create_task

# from .schema import PCAParams
# from .tasks import run_pca

router = APIRouter()

# @router.post(
#     "/pca",
#     status_code=status.HTTP_201_CREATED,
#     response_model=TaskPostResult,
#     name="comments:create-comment-for-article"
# )
# def pca(params: PCAParams):
#     res = create_task(run_pca, params.dict())
#     return TaskPostResult(**res)