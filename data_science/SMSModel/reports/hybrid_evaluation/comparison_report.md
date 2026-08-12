# Stacking·Claude·Hybrid 평가 보고서

## 평가 조건

- 생성 시각: `2026-08-12T16:01:42.931379+00:00`
- 평가 split: `test`
- 샘플 수: `123`
- 데이터셋 fingerprint: `508a702f225be77d2c8725d8f7f90a360f18f73a5af63815fd5d1513abc593bb`
- LLM: `AWS_BEDROCK` / `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- Region: `us-east-1`
- 프롬프트 버전: `smishing-v1:92102fecd96463d0d061cdf868df78f33de19ef86e44d508aae1d6c91897dee1`
- Stacking artifact SHA-256: `fa99576f1d655625410ddb3b6b36ab55faca5a780f296dae9041e37593c063fe`
- Stacking artifact schema: `1`

## 라우팅 정책

- 정책 선정 split: `validation`
- 정상 확신 상한: `0.08716666`
- 피싱 확신 하한: `0.59000000`
- Validation 예상 LLM 호출률: `41.46%`
- Test 실제 불확실 구간: `53/123` (`43.09%`)

## 모드별 비교

| 지표 | Stacking only | Claude only | Hybrid |
|---|---:|---:|---:|
| Accuracy | 0.6992 | 0.8537 | 0.8699 |
| Precision | 0.6442 | 0.7882 | 0.8072 |
| Recall | 1.0000 | 1.0000 | 1.0000 |
| F1 | 0.7836 | 0.8816 | 0.8933 |
| F2 | 0.9005 | 0.9490 | 0.9544 |
| 평균 지연시간 (ms) | 68.18 | 3234.83 | 1521.42 |
| P50 지연시간 (ms) | 24.25 | 3349.02 | 39.23 |
| P95 지연시간 (ms) | 91.83 | 5270.34 | 5376.85 |
| LLM 호출률 | 0.00% | 100.00% | 43.09% |
| 입력 token | 0 | 63444 | 36094 |
| 출력 token | 0 | 34406 | 15762 |
| 메시지당 비용 | 0.00000000 | 0.00191442 | 0.00093418 |
| 미측정 LLM 호출 | 0 | 0 | 0 |
| Fallback | 0 | 0 | 0 |
| 전체 엔진 실패 | 0 | 0 | 0 |
| 결과 없음 | 0 | 0 | 0 |

## Hybrid 개선 효과

- Claude-only 대비 LLM 호출 감소율: `56.91%`
- Claude-only 대비 메시지당 비용 감소율: `51.20%`
- Claude-only 대비 평균 지연시간 감소율: `52.97%`
- Claude-only 대비 P95 지연시간 감소율: `-2.02%`
- Stacking-only 대비 Recall 변화: `+0.0000`
- Stacking-only 대비 F2 변화: `+0.0539`

## 목표 지표 충족 여부

| 목표 | 기준 | 실제 | 결과 |
|---|---:|---:|:---:|
| Hybrid Recall | >= 0.9500 | 1.0000 | 충족 |
| Hybrid F2 vs Stacking-only | >= 0 delta | 0.0539 | 충족 |
| Claude call reduction | > 0% | 0.5691 | 충족 |
| Cost reduction | > 0% | 0.5120 | 충족 |
| Average latency reduction | > 0% | 0.5297 | 충족 |
| P95 latency reduction | > 0% | -0.0202 | 미충족 |

- 충족: `5/6`
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

`UNKNOWN` 결과는 정상으로 간주하지 않습니다. 이진 분류 지표에서 제외하고 각 모드의 `결과 없음` 건수로 별도 기록합니다.

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

## Issue #37 PR 3 체크리스트 (Bedrock/Claude)

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
