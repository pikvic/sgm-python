from typing import Optional
from starlette import status
from fastapi import APIRouter
from fastapi import Request

from app.core.schema import TaskPostResult, MethodInfo
from app.core.queue import create_task
from .schema import KMeansTaskParams, KMeansScreePlotTaskParams
from .tasks import run_kmeans, run_kmeansscreeplot
from app.methods.methods import METHODS


router = APIRouter()

@router.post(
    "/kmeans",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def kmeans_post(params: KMeansTaskParams, request: Request):
    res = create_task(run_kmeans, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/kmeans"
)
def kmeans_get():
    return METHODS['kmeans']

@router.post(
    "/kmeansscreeplot",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def kmeansscreeplot_post(params: KMeansScreePlotTaskParams, request: Request):
    res = create_task(run_kmeansscreeplot, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/kmeansscreeplot"
)
def kmeansscreeplot_get():
    return METHODS['kmeansscreeplot']

