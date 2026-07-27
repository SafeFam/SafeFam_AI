from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.analysis import router as analyze


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
    )

    # 기존 클라이언트 계약을 유지한다. 내부 서비스 전용 전환 시 CORS 제거를 검토한다.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(analyze.router, prefix="/api")

    @application.get("/", tags=["Root"])
    def root_check():
        return {"status": "healthy", "project": settings.PROJECT_NAME}

    return application


app = create_app()
