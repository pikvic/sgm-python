from typing import Optional
from starlette import status
from fastapi import APIRouter
from fastapi import Request

from app.core.schema import TaskPostResult, MethodInfo
from app.core.queue import create_task
from .schema import MissingValuesTaskParams, NormalizationTaskParams, StandartizationTaskParams
from .tasks import run_normalization, run_missingvalues, run_standartization
from app.methods.methods import METHODS


router = APIRouter()

@router.post(
    "/standartization",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def standartization_post(params: StandartizationTaskParams, request: Request):
    res = create_task(run_standartization, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/standartization"
)
def standartization_get():
    return METHODS['standartization']

@router.post(
    "/missingvalues",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def missingvalues_post(params: MissingValuesTaskParams, request: Request):
    res = create_task(run_missingvalues, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/missingvalues"
)
def missingvalues_get():
    return METHODS['missingvalues']

@router.post(
    "/normalization",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def normalization_post(params: NormalizationTaskParams, request: Request):
    res = create_task(run_normalization, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/normalization"
)
def normalization_get():
    return METHODS['normalization']