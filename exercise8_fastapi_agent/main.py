"""
Exercise 8 - Campus Query Router as a FastAPI service

Run:   uvicorn main:app --reload
Try:   POST http://127.0.0.1:8000/query   {"question": "When will the hall tickets be released?"}
Docs:  http://127.0.0.1:8000/docs
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router_agent import route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("campus-api")

app = FastAPI(title="Campus Query Router API")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


@app.post("/query")
def query(request: QueryRequest):
    log.info("REQUEST  question=%r", request.question)
    start = time.perf_counter()

    try:
        intent, answer = route(request.question)
    except Exception as e:
        log.error("FAILED   question=%r error=%s", request.question, e)
        raise HTTPException(status_code=502, detail="The agent could not process the question right now.")

    latency_ms = round((time.perf_counter() - start) * 1000)
    response = {
        "question": request.question,
        "intent": intent,
        "answer": answer,
        "latency_ms": latency_ms,
    }
    log.info("RESPONSE intent=%s latency=%dms answer=%r", intent, latency_ms, answer)
    return response
