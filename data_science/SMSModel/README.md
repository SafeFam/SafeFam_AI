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

## 실행 방법

프로젝트 루트 디렉토리에서 아래 명령어로 학습을 실행합니다.

```bash
python -m data_science.SMSModel.train_sms
```

커밋된 split manifest는 최종 test set을 고정합니다. 일반적인 재학습에서는
덮어쓰지 말고, 데이터셋·전처리·그룹화·분할 정책이 의도적으로 변경될 때만
새 manifest 버전을 생성하세요.

## Issue #37 PR 3: Bedrock Claude 하이브리드 평가 재현

평가 임계값은 validation split에서만 선정하며, 최종 비교에는 고정된 test
split만 사용합니다. 기본 실행은 저장된 Claude test 캐시만 읽으므로 AWS 호출과
추가 비용이 발생하지 않습니다.

```powershell
& .\.venv\Scripts\python.exe -m `
  data_science.SMSModel.run_hybrid_evaluation --offline

& .\.venv\Scripts\python.exe -m `
  data_science.SMSModel.generate_hybrid_evaluation_report `
  --input-price-per-million 1.0 `
  --output-price-per-million 5.0 `
  --currency USD `
  --pricing-as-of 2026-08-12
```

macOS/Linux에서는 다음과 같이 실행합니다.

```bash
./.venv/bin/python -m data_science.SMSModel.run_hybrid_evaluation --offline
./.venv/bin/python -m data_science.SMSModel.generate_hybrid_evaluation_report \
  --input-price-per-million 1.0 \
  --output-price-per-million 5.0 \
  --currency USD \
  --pricing-as-of 2026-08-12
```

캐시에 누락되거나 실패한 test 예측만 AWS Bedrock Claude Haiku에서 다시
수집하려면 profile을 지정하고 `--collect`를 사용합니다. 성공한 기존 예측은
재호출하지 않습니다.

```powershell
$env:AWS_PROFILE = "safefam-dev"
& .\.venv\Scripts\python.exe -m `
  data_science.SMSModel.run_hybrid_evaluation --collect
```

```bash
AWS_PROFILE=safefam-dev \
  ./.venv/bin/python -m data_science.SMSModel.run_hybrid_evaluation --collect
```

최종 산출물은 `reports/hybrid_evaluation/`의 `evaluation_records.json`,
`comparison_report.json`, `comparison_report.csv`, `comparison_report.md`입니다.
평가 레코드와 보고서에는 문자 원문, 개인정보, API key 또는 AWS 자격 증명을
저장하지 않습니다. 가격은 코드에 하드코딩하지 않으며 보고서 생성 인자로
전달한 가정값입니다.
