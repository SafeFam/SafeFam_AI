# SafeFam_AI
공경진 SafeFam AI Repository

## 프로젝트 구조

```
📁 SafeFam_AI
├── 📁 app
│   ├── 📁 core          # 환경 변수(Pydantic Settings) 및 글로벌 설정
│   ├── 📁 dto           # 데이터 전송 객체 (Spring 백엔드 통신 규격 통일)
│   ├── 📁 router        # 엔드포인트 라우터 레이어
│   └── 📄 main.py       # FastAPI 애플리케이션 진입점
├── 📁 data_science       # AI 모델 학습 및 데이터 관리 레이어 
├── 📄 .env              # 로컬 환경 변수 시크릿 키 (git 제외)
├── 📄 Dockerfile        # 파이썬 3.11기반 애플리케이션 이미지 빌드 정의
├── 📄 docker-compose.yml# 로컬 개발용 멀티 컨테이너 오케스트레이션 설정
└── 📄 requirements.txt  # 프로젝트 의존성 패키지 명세 목록
```

## 로컬 개발 환경 세팅
본 프로젝트는 로컬 개발 환경과 실제 배포 인프라 환경의 동기화를 위해 **Docker 컨테이너 기반 구동을 원칙**으로 합니다.

도커가 독립된 가상 컨테이너(Python 3.11 환경)를 알아서 띄워 그 안에서 서버를 실행합니다. 컨테이너 내부 핫 리로드(`--reload`)가 적용되어 있어 로컬 소스코드를 수정하면 실시간으로 반영됩니다.

---

### 1단계: Docker Desktop 설치하기

각자 운영체제(OS)에 맞는 Docker Desktop을 설치해 주세요.

1. **[Docker Desktop 다운로드 페이지](https://www.docker.com/products/docker-desktop/)**에 접속합니다.
2. 본인의 OS(Windows 또는 Mac Intel/Apple Silicon)에 맞는 버전을 다운로드하고 설치합니다.
3. **[Windows 사용자 필수 확인]** 
    - 설치 과정에서 `Use WSL 2 instead of Hyper-V` 체크박스가 나오면 **반드시 체크**하고 설치해주세요.
    - 설치 완료 후 컴퓨터를 재부팅하고 Docker Desktop을 실행했을 때, WSL 관련 업데이트 팝업이 뜨면 링크를 눌러 업데이트를 진행해 주셔야 정상 작동합니다.
4. Docker Desktop 프로그램을 켜두고 다음 단계로 진행합니다. (고래 모양 아이콘이 초록색 `Engine Running` 상태여야 합니다.)

---

### 2단계: 환경 변수 파일 생성
 
 프로젝트 최상위 루트 디렉토리(`SafeFam_AI/`)에 `.env` 파일을 생성하고 아래 내용을 입력합니다.

 ```env
 ENV=local
 VIRUSTOTAL_API_KEY=your_actual_api_key_here
 LLM_PROVIDER=bedrock
 AWS_REGION=us-east-1
 AWS_PROFILE=safefam-dev
 BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
 ```

---

### 3단계: Docker Compose 빌드 및 구동

터미널(VS Code 터미널 추천)을 열고 프로젝트 최상위 루트에서 아래 명령어를 입력합니다.

```bash
# 컨테이너 빌드 및 서버 실행 (최초 실행 시 빌드 시간이 약간 소요됩니다.)
docker compose up --build
```
- Tip1 (백그라운드 실행): 로그 대기 화면 없이 백그라운드에서 실행하고 싶다면 `-d` 옵션을 추가하세요: `docker compose up -d --build`
- Tip2 (단순 재구동): 코드가 이미 빌드된 상태에서 서버를 다시 켤 때는 `--build` 없이도 쳐도 됩니다: `docker compose up`

---

### 4단계: 정상 구동 확인
서버가 켜지면 브라우저를 열고 아래 주소로 접속하여 정상 작동하는지 확인합니다.

- **Swagger UI (API 문서)**: http://127.0.0.1:8000/docs

## 운영 환경

운영 환경에서는 `docker-compose.prod.yml`과 Git Commit SHA로 고정된 이미지를
사용합니다. `--reload`, 소스 코드 바인드 마운트, 호스트 포트 공개는 사용하지
않습니다.

```bash
cp .env.prod.example .env.runtime
docker compose -f docker-compose.prod.yml --env-file .env.runtime up -d
```

`.env.runtime`의 실제 값은 저장소에 커밋하지 않습니다. EC2 IAM Role로 AWS
Parameter Store의 `SecureString`을 조회하여 배포 시점에 생성합니다.

### 운영 필수 Secret

- `VIRUSTOTAL_API_KEY`
- `GOOGLE_SAFE_BROWSING_API_KEY`
- `RABBITMQ_URL`

권장 Parameter Store 경로는 다음과 같습니다.

```text
/safefam/prod/ai/VIRUSTOTAL_API_KEY
/safefam/prod/ai/GOOGLE_SAFE_BROWSING_API_KEY
```

운영에서 필수 Secret이 누락되거나 `MOCK_SECURITY_API=true`이면 애플리케이션은
시작하지 않습니다.

### 모델 파일

운영 이미지에는 아래 두 개의 검증된 학습 산출물만 포함합니다.

```text
/app/models/phishing_model_artifact.pkl
/app/models/phishing_vectorizer.pkl
```

컨테이너 시작 시 두 파일이 없으면 애플리케이션이 즉시 실패합니다. Pickle은
임의 파일을 실행할 위험이 있으므로 저장소에서 관리하는 신뢰된 산출물만
사용해야 합니다.
