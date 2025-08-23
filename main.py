from fastapi import FastAPI, HTTPException
import traceback
from dotenv import load_dotenv

from schemas.schema import QueryIn
from log import log_and_save_response, log_incoming_request

from rag_application.web import web

load_dotenv(verbose=True)
app = FastAPI()

from rag_application.web import web

@app.get("/")
async def root():
    print("in root")
    return {"message": "Welcome", "status": "up"}

@app.post("/hackrx/run")
async def protected_route(queryIn: QueryIn):
    log_incoming_request({
        "url": queryIn.url,
        "document_url": queryIn.document_url,
        
        "questions": queryIn.questions
    })

    # Validate input: exactly one of url or document_url must be provided
    if (queryIn.url and queryIn.document_url) or (not queryIn.url and not queryIn.document_url):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of 'url' or 'document_url' must be provided."
        )

    sources = []
    if queryIn.url:
        sources.append(queryIn.url)
    elif queryIn.document_url:
        sources.append(queryIn.document_url)

    try:
        answers = await web({"url" : sources[0], "questions" : queryIn.questions})
        return {"answers": [q["answer"] for q in answers.values()]}
    except Exception as e:
        error_data = {
            "error": str(e),
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        log_and_save_response(error_data, False)
        raise HTTPException(status_code=500, detail=str(e))