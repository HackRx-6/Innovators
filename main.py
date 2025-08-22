import os
import time
import traceback

from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from starlette.middleware.cors import CORSMiddleware
from sympy import false
import httpx


from rag_application.run_pipeline import run_pipeline
from schemas.schema import QueryIn
from log import log_and_save_response, log_incoming_request

load_dotenv(verbose=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
async def root():
    return {"message": "Welcome", "status": "up"}


@app.post("/hackrx/run", )
async def protected_route(queryIn: QueryIn):
    log_incoming_request({"documents": queryIn.documents, "questions": queryIn.questions})

    try:
        answers = await run_pipeline(queryIn.documents.strip(), queryIn.questions)
        return {"answers": answers}
    except Exception as e:
        error_data = {
            "error": str(e),
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        log_and_save_response(error_data, false)
        raise HTTPException(status_code=500, detail=str(e))
