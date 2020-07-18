from fastapi import FastAPI, HTTPException, status, Request, APIRouter
from fastapi.responses import FileResponse
from rq.job import Job

import app.core.config as config
from app.core.schema import TaskResult
from app.core.queue import get_queue, get_redis

router = APIRouter()

@router.get(
    "/",
    summary="Проверка работы узла",
    description="Возвращает Hello, World!, если работает."
)
def root():
    return {"message": "Hello World!"}

@router.get(
    "/jobs",
    summary="Получение списка выполненных задач",
    description="Возвращает список выполненных задач."
)
def jobs_list():
    queue = get_queue()
    job_ids = queue.finished_job_registry.get_job_ids()
    jobs = []
    for job_id in job_ids:
        job = queue.fetch_job(job_id)
        jobs.append({'id': job.id, 'funcname': job.func_name, 'status': job.get_status()})
    res = {'jobs': jobs}
    return res

@router.get(
    "/results/{job_id}", 
    response_model=TaskResult,
    summary="Получение результата по задаче",
    description="Возвращает состояние задачи и результат, если есть."
)
def get_result(job_id):
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Task not found!')
    if job.is_finished:
        res = job.result
        if 'error' in res:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=res['error'])
        return TaskResult(**res)
    else:
        return TaskResult(ready=False)


@router.get(
    "/download/{job_id}/{filename}",
    summary="Скачивание файла с результатом по задаче",
    description="Возвращает конкретный файл с результатом."
)
def get_file(job_id, filename):
    path = config.DOWNLOAD_DIR / job_id / filename
    if path.exists():
        return FileResponse(str(path))
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='File not found!')

