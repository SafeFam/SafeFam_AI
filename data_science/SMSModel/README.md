# SMS 모델 워크스페이스

| 경로 | 설명 |
|---|---|
| `train_sms.py` | Naive Bayes 모델 학습 엔트리포인트 |
| `run_naive_bayes_baseline.py` | Naive Bayes 베이스라인 파이프라인 실행 및 평가 스크립트 |
| `artifacts/` | API에서 사용하는 버전 관리된 모델 및 벡터라이저 파일 |
| `tokenization/` | 형태소 및 문자 기반 토큰화 처리 패키지 |
| `modeling/` | 피싱 분류 모델 어댑터 및 인터페이스 |
| `evaluation/` | 모델 평가 및 메트릭 산출 패키지 |
| `dataset_splitting/` | 데이터 누수(Leakage)를 방지하는 train/validation/test 데이터 분할 |
| `template_grouping/` | 중복 및 유사 문자 그룹화 |
| `splits/` | 재현 가능한 데이터 분할 매니페스트 |
| `reporting/` | 데이터셋 보고서 생성 코드 |
| `reports/` | 생성된 데이터 요약, 평가 메트릭 및 피처 분석 결과 |
| `reports/figures/` | 생성된 그래프 및 시각화 차트 |
| `SMSDataModel.ipynb` | 데이터 탐색 및 분석용 주피터 노트북 |

---

### 실행 방법

프로젝트 루트 디렉토리에서 아래 명령어로 학습을 실행합니다.

```bash
python -m data_science.SMSModel.train_sms