from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from app.core.config import settings
from app.router import analyze

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# Spring Boot 및 프론트엔드 연동을 위한 CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 통합 등록 
app.include_router(analyze.router, prefix="/api")

@app.get("/", tags=["Root"])
def root_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}