from fastapi import FastAPI
from app.core.config import settings
from app.router import analyze

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

app.include_router(analyze.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}