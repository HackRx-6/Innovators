from typing import List, Optional
from pydantic import BaseModel, model_validator

class QueryIn(BaseModel):
    url: Optional[str] = None
    document_url: Optional[str] = None
    questions: List[str]

    @model_validator(mode="after")
    def check_one_present(cls, values):
        url, document_url = values.url, values.document_url
        if (url and document_url) or (not url and not document_url):
            raise ValueError("Exactly one of 'url' or 'document_url' must be provided.")
        return values