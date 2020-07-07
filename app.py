from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl, AnyUrl
from typing import Optional, List

from datetime import timedelta
from pathlib import Path
from redis import from_url
from rq import Queue
from rq.job import Job
from uuid import uuid4

from tasks import run_stats, clear_files_for_job, run_kmeans, run_hca, run_linear, run_pca
import config


class StatsTaskParams(BaseModel):
    url: HttpUrl
    column: int = Field(..., gt=0)
    transpose: Optional[bool] = False
    showgraph: Optional[bool] = False

class KMeansParams(BaseModel):
    url: HttpUrl
    nclusters: int = 6
    randomstate: Optional[int] = 0
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    addresultcolumns: Optional[bool] = False
    showstats: Optional[bool] = False
    normalize: Optional[bool] = False
    showgraph: Optional[bool] = False

class PCAParams(BaseModel):
    url: HttpUrl
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    normalize: Optional[bool] = False
    showgraph: Optional[bool] = False

class HCAParams(BaseModel):
    url: HttpUrl
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    normalize: Optional[bool] = False

class LinearParams(BaseModel):
    url: HttpUrl
    xcolumn: int = Field(..., gt=0)
    ycolumn: int = Field(..., gt=0)

class TaskPostResult(BaseModel):
    job_id: str
    url: str

class TaskResult(BaseModel):
    ready: bool = False
    results: List[str] = None



app = FastAPI()
red = from_url(config.REDIS_URL)
queue = Queue(connection=red, default_timeout=3600)

if not config.UPLOAD_DIR.exists():
    config.UPLOAD_DIR.mkdir()
if not config.DOWNLOAD_DIR.exists():
    config.DOWNLOAD_DIR.mkdir()

print(config.UPLOAD_DIR)
print(config.DOWNLOAD_DIR)
print([str(d) for d in config.ROOT.iterdir()])

def generate_id():
    return str(uuid4())

@app.get("/")
def root():
    return {"message": "Hello World!", "uuid": generate_id()}

@app.get("/jobs")
def jobs_list():
    #jobs = queue.jobs
    job_ids = queue.finished_job_registry.get_job_ids()
    #queue.fetch_job(job_id)
    jobs = []
    for job_id in job_ids:
        job = queue.fetch_job(job_id)
        jobs.append({'id': job.id, 'funcname': job.func_name, 'status': job.get_status()})
    res = {'jobs': jobs}
    return res

@app.get("/results/{job_id}", response_model=TaskResult)
def get_result(job_id):
    try:
        job = Job.fetch(job_id, connection=red)
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Task not found!')
    if job.is_finished:
        res = job.result
        if 'error' in res:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=res['error'])
        return TaskResult(**res)
    else:
        return TaskResult(ready=False)


@app.get("/download/{job_id}/{filename}")
def get_file(job_id, filename):
    path = config.DOWNLOAD_DIR / job_id / filename
    if path.exists():
        return FileResponse(str(path))
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='File not found!')

@app.post("/stats", response_model=TaskPostResult)
def stats(params: StatsTaskParams):
    params_dict = params.dict()
    job_id = generate_id()
    params_dict['job_id'] = job_id
    job = queue.enqueue_call(func=run_stats, args=(params_dict,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,))
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return TaskPostResult(**res)

@app.post('/kmeans', response_model=TaskPostResult)
def kmeans(params: KMeansParams):
    params_dict = params.dict()
    job_id = generate_id()
    params_dict['job_id'] = job_id
    job = queue.enqueue_call(func=run_kmeans, args=(params_dict,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,))
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return TaskPostResult(**res)
    
@app.post('/pca', response_model=TaskPostResult)
def pca(params: PCAParams):
    params_dict = params.dict()
    job_id = generate_id()
    params_dict['job_id'] = job_id
    job = queue.enqueue_call(func=run_pca, args=(params_dict,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,))
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return TaskPostResult(**res)
    
@app.post('/hca', response_model=TaskPostResult)
def hca(params: HCAParams):
    params_dict = params.dict()
    job_id = generate_id()
    params_dict['job_id'] = job_id
    job = queue.enqueue_call(func=run_hca, args=(params_dict,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,))
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return TaskPostResult(**res)
    
@app.post('/linear', response_model=TaskPostResult)
def linear(params: LinearParams):
    params_dict = params.dict()
    job_id = generate_id()
    params_dict['job_id'] = job_id
    job = queue.enqueue_call(func=run_linear, args=(params_dict,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,))
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return TaskPostResult(**res)