# Stacking·Claude·Hybrid 평가 보고서

## 평가 조건

- 생성 시각: `2026-08-16T07:41:09.640776+00:00`
- 평가 split: `test`
- 샘플 수: `126`
- 라벨 분포: `normal=51`, `phishing=75`
- Positive label: `phishing`
- 데이터셋 fingerprint: `d10c1de722b2caa9f50e3d3873c9afaab7d9472fca304f1c5cd91390b0764d6f`
- Split manifest: `sms_split_v1.csv`
- Split manifest SHA-256: `4c33fcc749ca5c8ac4e6b35d19831e3ad8bffb02e72ac4d50d423d7ca19aa5f9`
- Random state: `42`
- LLM: `AWS_BEDROCK` / `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- Region: `us-east-1`
- 프롬프트 버전: `smishing-v1:92102fecd96463d0d061cdf868df78f33de19ef86e44d508aae1d6c91897dee1`
- Stacking artifact SHA-256: `fa99576f1d655625410ddb3b6b36ab55faca5a780f296dae9041e37593c063fe`
- Stacking artifact schema: `1`
- Stacking classification threshold: `0.08844760`
- Threshold source split: `validation`
- Threshold selection: `maximize_f2_subject_to_target_recall`

## 라우팅 정책

- 정책 선정 split: `validation`
- 정상 확신 상한: `0.08716666`
- 피싱 확신 하한: `0.59000000`
- Validation 예상 LLM 호출률: `41.46%`
- Test 실제 불확실 구간: `29/126` (`23.02%`)

## 전체 표본 기준 비교

`UNKNOWN`을 포함한 전체 test 표본을 분모로 사용합니다. 사용 불가능한 결과는 전체 정확도에서 실패로, 실제 피싱 표본에서는 미탐으로 계산합니다.

| 지표 | Stacking only | Claude only | Hybrid |
|---|---:|---:|---:|
| 전체 표본 수 | 126 | 126 | 126 |
| 결과 있음 | 126 | 104 | 126 |
| 결과 없음 | 0 | 22 | 0 |
| Availability | 100.00% | 82.54% | 100.00% |
| 전체 정확도 | 0.7857 | 0.7381 | 0.8730 |
| 피싱 탐지율 | 1.0000 | 0.8800 | 1.0000 |
| 정답 | 99 | 93 | 110 |
| 오분류 | 27 | 11 | 16 |
| 피싱 미탐 | 0 | 9 | 0 |

## 결과 성공 표본 기준 이진 분류

아래 지표는 `UNKNOWN`을 제외한 결과이므로 Availability와 함께 해석해야 합니다.

| 지표 | Stacking only | Claude only | Hybrid |
|---|---:|---:|---:|
| 평가 표본 수 | 126 | 104 | 126 |
| Accuracy | 0.7857 | 0.8942 | 0.8730 |
| Precision | 0.7353 | 0.8571 | 0.8242 |
| Recall | 1.0000 | 1.0000 | 1.0000 |
| F1 | 0.8475 | 0.9231 | 0.9036 |
| F2 | 0.9328 | 0.9677 | 0.9591 |
| TN | 24 | 27 | 35 |
| FP | 27 | 11 | 16 |
| FN | 0 | 0 | 0 |
| TP | 75 | 66 | 75 |

## 운영·비용 비교

Latency는 성공·실패·fallback을 모두 포함한 메시지 단위 end-to-end 처리 시간입니다.

| 지표 | Stacking only | Claude only | Hybrid |
|---|---:|---:|---:|
| 평균 지연시간 (ms) | 52.96 | 2409.79 | 462.67 |
| P50 지연시간 (ms) | 26.40 | 3075.04 | 28.35 |
| P95 지연시간 (ms) | 43.75 | 3771.06 | 3214.16 |
| LLM 호출률 | 0.00% | 100.00% | 23.02% |
| 입력 token | 0 | 42508 | 8342 |
| 출력 token | 0 | 27745 | 4626 |
| 메시지당 비용 | 0.00000000 | 0.00143836 | 0.00024978 |
| 미측정 LLM 호출 | 0 | 22 | 8 |
| Fallback | 0 | 0 | 8 |
| 전체 엔진 실패 | 0 | 22 | 0 |

## Hybrid 개선 효과

- Claude-only 대비 LLM 호출 감소율: `76.98%`
- Claude-only 대비 메시지당 비용 감소율: `82.63%`
- Claude-only 대비 평균 지연시간 감소율: `80.80%`
- Claude-only 대비 P95 지연시간 감소율: `14.77%`
- Stacking-only 대비 Recall 변화: `+0.0000`
- Stacking-only 대비 F2 변화: `+0.0262`

## 목표 지표 충족 여부

| 목표 | 기준 | 실제 | 결과 |
|---|---:|---:|:---:|
| Hybrid full-dataset phishing detection rate | >= 0.9500 | 1.0000 | 충족 |
| Hybrid F2 vs Stacking-only | >= 0 delta | 0.0262 | 충족 |
| Claude call reduction | > 0% | 0.7698 | 충족 |
| Cost reduction | > 0% | 0.8263 | 충족 |
| Average latency reduction | > 0% | 0.8080 | 충족 |
| P95 latency reduction | > 0% | 0.1477 | 충족 |

- 충족: `6/6`
- P95 지연시간 감소 목표는 미충족이며 결과를 그대로 기록했습니다.

## 임계값 채택 결론

- 상태: `ADOPTED`
- 근거: Validation-selected thresholds preserve the target recall on the untouched test split, improve F2 over Stacking-only, and reduce Claude calls, cost, and average latency. The P95 latency target was not met and is recorded as a follow-up limitation.

## 비용 가정

- 통화: `USD`
- 가격 기준일: `2026-08-13`
- 입력 token 100만 개당 가격: `1.0`
- 출력 token 100만 개당 가격: `5.0`

## 판정 정책

`UNKNOWN` 결과는 정상으로 간주하지 않습니다. Available-only 이진 분류 지표에서는 제외하지만 전체 표본 지표의 분모에는 유지하며, 실제 피싱의 `UNKNOWN`은 미탐으로 계산합니다.

## 재현 명령어

프로젝트 루트에서 실행합니다. 기본 실행은 Bedrock을 호출하지 않는 오프라인 캐시 재평가입니다.

```powershell
& .\.venv\Scripts\python.exe -m data_science.SMSModel.run_hybrid_evaluation --offline
& .\.venv\Scripts\python.exe -m data_science.SMSModel.generate_hybrid_evaluation_report `
  --input-price-per-million 1.0 `
  --output-price-per-million 5.0 `
  --currency USD `
  --pricing-as-of 2026-08-13
```

캐시에 누락된 test 예측만 Bedrock에서 수집할 때는 AWS profile을 설정한 뒤 `--collect`를 사용합니다.

```powershell
$env:AWS_PROFILE = "safefam-dev"
& .\.venv\Scripts\python.exe -m data_science.SMSModel.run_hybrid_evaluation --collect
```

## 평가 검증 체크리스트

- [x] 고정된 test split에서 Stacking-only 평가
- [x] AWS Bedrock Claude Haiku test 예측 수집
- [x] Claude-only 및 Hybrid 평가
- [x] validation 전용 임계값 선정 및 test 재조정 방지
- [x] 정상·실패·fallback·하위 호환성 통합 테스트
- [x] 성능·호출률·지연시간·token·비용 비교
- [x] JSON·CSV·Markdown 결과 생성
- [x] 원문·개인정보·AWS 자격 증명 미저장
- [x] 목표 미충족 결과(P95 지연시간) 공개
- [x] 임계값 채택 여부와 재현 방법 문서화
