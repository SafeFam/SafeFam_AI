from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from app.core.config import settings
from app.router import analyze

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}