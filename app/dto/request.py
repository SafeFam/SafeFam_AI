from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    message: str
