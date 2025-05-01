from fastapi import APIRouter, HTTPException
import redis
import json
import os
import datetime
import traceback
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
REDIS_CONNECT_TIMEOUT = int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
REDIS_SOCKET_TIMEOUT = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
REDIS_JOB_EXPIRY_SECONDS = int(os.getenv('REDIS_JOB_EXPIRY_SECONDS', '86400'))

try:
    redis_client = redis.Redis.from_url(
        REDIS_URL,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        decode_responses=True
    )
    redis_client.ping()
    print(f"Progress API connected to Redis at: {REDIS_URL}")
except Exception as e:
    print(f"FATAL: Failed to connect to Redis at {REDIS_URL}: {e}")
    redis_client = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    percent_complete: int = 0
    filename: Optional[str] = None
    current_page: Optional[int] = 0
    total_pages: Optional[int] = 0

EXTRACTION_START_PERC = 0
EXTRACTION_END_PERC = 69

@router.get("/{job_id}", response_model=JobStatus)
async def get_progress(job_id: str):
    if not redis_client:
        print(f"ERROR: get_progress called but Redis client is not initialized.")
        raise HTTPException(status_code=503, detail="Backend Redis connection failed")

    redis_key = f"job:{job_id}"
    current_dt = datetime.datetime.now().isoformat()

    try:
        job_data = redis_client.get(redis_key)

        if not job_data:
            print(f"[{current_dt}] Progress request: Key '{redis_key}' not found in Redis.")
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        try:
            job = json.loads(job_data)
            return JobStatus(**job)
        except json.JSONDecodeError as json_err:
            print(f"[{current_dt}] ERROR decoding job data for {job_id}. Data: '{job_data}'. Error: {json_err}")
            raise HTTPException(status_code=500, detail="Invalid job data in Redis")
        except Exception as pydantic_err:
             print(f"[{current_dt}] ERROR validating job data for {job_id}. Data: '{job_data}'. Error: {pydantic_err}")
             raise HTTPException(status_code=500, detail="Job data validation error")

    except redis.exceptions.ConnectionError as e:
        print(f"[{current_dt}] ERROR: Redis ConnectionError in get_progress for {job_id}: {e}")
        raise HTTPException(status_code=503, detail=f"Redis connection error: {e}")
    except HTTPException:
         raise
    except Exception as e:
        print(f"[{current_dt}] ERROR: Unexpected error in get_progress for {job_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")

def create_job(job_id: str, filename: str, total_pages: int) -> None:
    if not redis_client:
        print(f"ERROR: create_job called but Redis client is not initialized.")
        return
    try:
        job = {
            "job_id": job_id,
            "filename": filename,
            "current_page": 0,
            "total_pages": total_pages,
            "percent_complete": 0,
            "status": "starting",
            "message": "Initializing..."
        }
        redis_client.setex(f"job:{job_id}", REDIS_JOB_EXPIRY_SECONDS, json.dumps(job))
        print(f"Created job: {job_id} for file {filename} with {total_pages} pages")
    except Exception as e:
        print(f"ERROR in create_job: {type(e).__name__}: {e}")

async def update_job_status(job_id: str, status: str, message: str, percent_complete: Optional[int] = None, current_page: Optional[int] = None) -> None:
    if not redis_client:
        print(f"ERROR: update_job_status called but Redis client is not initialized.")
        return
    try:
        redis_key = f"job:{job_id}"
        job_data = redis_client.get(redis_key)
        if job_data:
            try:
                job = json.loads(job_data)
                job["status"] = status
                job["message"] = message
                if percent_complete is not None:
                    job["percent_complete"] = max(0, min(100, percent_complete))
                if current_page is not None:
                     job["current_page"] = current_page

                redis_client.setex(redis_key, REDIS_JOB_EXPIRY_SECONDS, json.dumps(job))
            except Exception as e:
                print(f"ERROR updating job status fields for {job_id}: {e}")
    except Exception as e:
        print(f"ERROR retrieving job for status update {job_id}: {e}")

async def update_job_progress(job_id: str, current_page: int) -> None:
    if not redis_client:
        print(f"ERROR: update_job_progress called but Redis client is not initialized.")
        return
    try:
        redis_key = f"job:{job_id}"
        job_data = redis_client.get(redis_key)

        if job_data:
            try:
                job = json.loads(job_data)
                if job.get("status") not in ["completed", "failed", "error"]:
                    job["current_page"] = current_page
                    total_pages = job.get("total_pages", 0)
                    raw_percent = 0
                    if total_pages > 0:
                         raw_percent = (current_page / total_pages) * 100

                    extraction_phase_range = EXTRACTION_END_PERC - EXTRACTION_START_PERC
                    scaled_percent = EXTRACTION_START_PERC + int((raw_percent / 100.0) * extraction_phase_range)

                    job["percent_complete"] = max(EXTRACTION_START_PERC, min(EXTRACTION_END_PERC, scaled_percent))
                    job["message"] = f"Extracting page {current_page} of {total_pages}"
                    job["status"] = "extracting"

                    redis_client.setex(redis_key, REDIS_JOB_EXPIRY_SECONDS, json.dumps(job))
            except Exception as e:
                print(f"ERROR in update_job_progress logic for {job_id}: {type(e).__name__}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"ERROR in update_job_progress connection/retrieval for {job_id}: {type(e).__name__}: {e}")
        traceback.print_exc()

async def progress_complete_job(job_id: str, message: str = "Processing complete", final_status: str = "completed") -> None:
    if not redis_client:
        print(f"ERROR: complete_job called but Redis client is not initialized.")
        return
    try:
        redis_key = f"job:{job_id}"
        job_data = redis_client.get(redis_key)
        job = {}

        if job_data:
            try:
                job = json.loads(job_data)
            except json.JSONDecodeError as e:
                print(f"ERROR: Could not parse existing job data in complete_job: {e}. Data: {job_data}")

        job["job_id"] = job_id
        job["status"] = final_status
        job["message"] = message
        if final_status == "completed":
            job["percent_complete"] = 100
            if "total_pages" in job:
                 job["current_page"] = job.get("total_pages")
        else:
            job["percent_complete"] = job.get("percent_complete", 0)

        redis_client.setex(redis_key, REDIS_JOB_EXPIRY_SECONDS, json.dumps(job))

        completion_dt = datetime.datetime.now().isoformat()
        print(f"[{completion_dt}] Finalized job: {job_id} with status '{final_status}' - {message}")

    except Exception as e:
        error_dt = datetime.datetime.now().isoformat()
        print(f"[{error_dt}] ERROR in complete_job for {job_id}: {type(e).__name__}: {e}")
        traceback.print_exc()

def complete_job_sync(job_id: str, message: str = "Processing complete", final_status: str = "completed") -> None:
    if not redis_client:
        print(f"ERROR: complete_job_sync called but Redis client is not initialized.")
        return
    try:
        redis_key = f"job:{job_id}"
        job_data = redis_client.get(redis_key)
        job = {}

        if job_data:
            try:
                job = json.loads(job_data)
            except json.JSONDecodeError as e:
                print(f"ERROR: Could not parse existing job data: {e}")

        job["job_id"] = job_id
        job["status"] = final_status
        job["message"] = message
        if final_status == "completed":
            job["percent_complete"] = 100
            if "total_pages" in job:
                job["current_page"] = job["total_pages"]
        else:
            job["percent_complete"] = job.get("percent_complete", 0)

        redis_client.setex(redis_key, REDIS_JOB_EXPIRY_SECONDS, json.dumps(job))

        completion_dt = datetime.datetime.now().isoformat()
        print(f"[{completion_dt}] Finalized job (sync): {job_id} with status '{final_status}' - {message}")

    except Exception as e:
        error_dt = datetime.datetime.now().isoformat()
        print(f"[{error_dt}] ERROR in complete_job_sync for {job_id}: {type(e).__name__}: {e}")
        traceback.print_exc()