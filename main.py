from fastapi import FastAPI, HTTPException
import traceback
from dotenv import load_dotenv

from schemas.schema import QueryIn
from log import log_and_save_response, log_incoming_request
from rag_application.new_web import main

load_dotenv(verbose=True)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome", "status": "up"}

def validate_input(queryIn: QueryIn):
    if (queryIn.url and queryIn.document_url) or (not queryIn.url and not queryIn.document_url):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of 'url' or 'document_url' must be provided."
        )
    return queryIn.url or queryIn.document_url

def extract_answers(answer_dict):
    return [v["answer"] for v in answer_dict.values()]

@app.post("/hackrx/run")
async def protected_route(queryIn: QueryIn):
    log_incoming_request({
        "url": queryIn.url,
        "document_url": queryIn.document_url,
        "questions": queryIn.questions
    })

    source = validate_input(queryIn)

    try:
        answer = await main(source, queryIn.questions)
        print("DEBUG: main returned:", answer)  # Add this line
        answers_list = extract_answers(answer)
        return {"answers": answers_list}
    except Exception as e:
        error_data = {
            "error": str(e),
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        log_and_save_response(error_data, False)
        raise HTTPException(status_code=500, detail="Internal server error")