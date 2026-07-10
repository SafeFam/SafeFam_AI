FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /workspace

# 필수 패키지 설치를 위한 레이어 캐싱
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 전체 복사
COPY . .

# FastAPI 기본 포트 개방
EXPOSE 8000

# 로컬 개발 및 컨테이너 구동을 위한 기본 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]