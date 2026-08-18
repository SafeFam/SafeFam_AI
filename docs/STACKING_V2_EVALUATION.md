# Stacking v2 재학습 및 성능 평가

## 1. 목적과 결론

이 문서는 GitHub Issue #78에서 수행한 Stacking v2 재학습 및 성능 검증 결과를 기록한다.

Stacking v2는 피싱 Recall `1.0`을 달성했지만, 독립 Test에서 정상 메시지 60건 중 59건을 피싱으로 분류했다. 공통 holdout에서도 v1 대비 정상 오탐이 1건 감소하는 데 그쳤으며, McNemar exact test 결과 통계적으로 유의한 개선이 아니었다.

**최종 결정: `REJECT`**

- Stacking v2를 단독 운영 모델로 채택하지 않는다.
- 기존 운영 artifact를 v2 artifact로 교체하지 않는다.
- v2 artifact는 실험 결과 재현과 후속 분석을 위해 별도 경로에 보존한다.
- 후속 실험에서는 validation 목표를 Recall 단독으로 두지 말고 정상 메시지 오탐 제한을 함께 적용해야 한다.

## 2. 데이터 및 분할 검증

### 2.1 데이터 구성

| 구분 | 표본 수 | 용도 |
|---|---:|---|
| 전체 CSV | 3,002 | 원본 데이터 전체 |
| Train/Validation/Test pool | 885 | 중복 제거 및 그룹화 후 학습·평가 대상 |
| Train | 623 | 모델 학습 전용 |
| Validation | 126 | 분류 threshold 선정 전용 |
| Test | 136 | v2 독립 성능 평가 전용 |
| 원래 holdout | 210 | v1/v2 공통 비교 후보 |
| Leak-free holdout | 190 | 실제 v1/v2 공정 비교 대상 |

- Dataset fingerprint: `46c1c9393d30f25ab03f0f7b6e85e5a8706f682f9bf2b68c872567b7eb5256f2`
- Split manifest: `sms_split_v2.csv`
- Split manifest SHA-256: `0281fc66f3f7ad288756bfcc0f768b7153f9dbdd61aeb3fca6d04316b4d5f49e`
- Train/Validation/Test 간 text fingerprint 중복: 0건
- Train/Validation/Test 간 template group 중복: 0건
- 기존 `sms_split_v1.csv`와 새 `sms_split_v2.csv`는 모두 보존한다.

### 2.2 Holdout 격리 감사

원래 holdout 210건을 검사한 결과, 다음 데이터가 Train pool과 동일한 text fingerprint를 사용하고 있었다.

- Train source: `synthetic_fp_stress_train`
- Holdout source: `synthetic_fp_stress`
- 중복 fingerprint 수: 1개
- 중복 holdout 행 수: 20건

이 20건을 포함하면 v2가 학습 중 본 메시지를 holdout에서 다시 평가하게 되므로 공정한 비교가 아니다. 데이터셋과 #77의 확정 fingerprint를 사후 변경하지 않고, 중복 20건을 비교 대상에서 제외했다.

따라서 공통 holdout 비교 결과는 210건 전체가 아니라 **학습 데이터와 fingerprint가 겹치지 않는 190건**을 기준으로 한다. 보고서에는 원래 표본 수, 실제 평가 표본 수, 제외 행 수와 제외 fingerprint 수를 함께 기록했다.

## 3. 학습 및 평가 정책

- 모델 구조와 전처리는 기존 Stacking 모델과 동일하게 유지했다.
- 학습 정답은 `normal/phishing` 이진 label만 사용했다.
- `type`은 성능 분석용 보조 label로만 사용했다.
- Random seed는 `42`로 고정했다.
- OOF split 수는 `5`로 고정했다.
- Train 623건만 모델 학습에 사용했다.
- Validation 126건에서 목표 피싱 Recall `0.95`를 만족하는 후보 중 F2가 가장 높은 threshold를 선택했다.
- 선택된 threshold를 고정한 후 Test 136건을 평가했다.
- Test 또는 holdout 결과를 이용해 threshold나 모델 설정을 다시 조정하지 않았다.
- Holdout은 v1/v2 비교에만 사용했다.

선택된 v2 threshold는 다음과 같다.

```text
0.014429172671629664
```

Validation 결과:

| Metric | Value |
|---|---:|
| Target Recall | 0.9500 |
| Recall | 1.0000 |
| F2 | 0.8483 |
| Target met | Yes |

## 4. 실행 환경

| 항목 | 버전 |
|---|---|
| Python | 3.13.14 |
| joblib | 1.5.3 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| SciPy | 1.18.0 |

## 5. v2 독립 Test 결과

이 결과는 `sms_split_v2.csv`의 Test 136건에 대한 v2 독립 평가다. v1의 기존 Test는 표본 구성이 다르므로 이 표와 직접 비교하지 않는다.

### 5.1 전체 지표

| Metric | Value |
|---|---:|
| 전체 표본 수 | 136 |
| 결과 산출 성공 | 136 |
| 결과 없음 | 0 |
| 예외 | 0 |
| 전체 엔진 실패 | 0 |
| Accuracy | 0.5662 |
| Precision | 0.5630 |
| Recall | 1.0000 |
| F1 | 0.7204 |
| F2 | 0.8656 |
| TN | 1 |
| FP | 59 |
| FN | 0 |
| TP | 76 |

피싱 메시지는 모두 탐지했지만 정상 메시지 60건 중 59건을 피싱으로 오탐했다. 정상 메시지 기준 오탐률은 `98.33%`다.

### 5.2 고유 template group 기준 결과

Test의 136개 행은 90개 고유 template group으로 구성된다. 각 그룹의 대표 표본을 기준으로 계산한 결과는 다음과 같다.

| Metric | Value |
|---|---:|
| 고유 template 수 | 90 |
| Accuracy | 0.4889 |
| Precision | 0.4831 |
| Recall | 1.0000 |
| F1 | 0.6515 |
| F2 | 0.8238 |
| TN | 1 |
| FP | 46 |
| FN | 0 |
| TP | 43 |

고유 template 기준으로도 정상 template 오탐 문제가 유지된다.

### 5.3 피싱 유형별 Recall

| 피싱 유형 | 표본 수 | TP | FN | Recall | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| 경조사사칭 | 9 | 9 | 0 | 1.0000 | 0.7008–1.0000 |
| 계정정지본인인증유도 | 1 | 1 | 0 | 1.0000 | 0.2065–1.0000 |
| 금융기관사칭 | 18 | 18 | 0 | 1.0000 | 0.8241–1.0000 |
| 대출사기 | 1 | 1 | 0 | 1.0000 | 0.2065–1.0000 |
| 악성링크앱설치유도 | 1 | 1 | 0 | 1.0000 | 0.2065–1.0000 |
| 이벤트당첨사칭 | 18 | 18 | 0 | 1.0000 | 0.8241–1.0000 |
| 정부공공기관사칭 | 3 | 3 | 0 | 1.0000 | 0.4385–1.0000 |
| 중고거래사기 | 1 | 1 | 0 | 1.0000 | 0.2065–1.0000 |
| 채용부업사기 | 15 | 15 | 0 | 1.0000 | 0.7961–1.0000 |
| 택배배송사칭 | 8 | 8 | 0 | 1.0000 | 0.6756–1.0000 |
| 투자리딩방사기 | 1 | 1 | 0 | 1.0000 | 0.2065–1.0000 |

모든 관측 유형에서 FN은 0건이었다. 다만 표본이 1~3건인 유형은 신뢰구간이 매우 넓으므로 해당 비율만으로 일반화할 수 없다.

### 5.4 정상 유형별 오탐

| 정상 유형 | 표본 수 | TN | FP | 오탐률 | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| 기타정상 | 36 | 1 | 35 | 0.9722 | 0.8583–0.9951 |
| 일상대화 | 22 | 0 | 22 | 1.0000 | 0.8513–1.0000 |
| 정상공공기관알림 | 1 | 0 | 1 | 1.0000 | 0.2065–1.0000 |
| 정상포인트소멸알림 | 1 | 0 | 1 | 1.0000 | 0.2065–1.0000 |

특히 일상대화 22건을 모두 피싱으로 판정했다. 이는 Stacking 단독 운영을 불가능하게 만드는 핵심 실패 요인이다.

### 5.5 추론시간

| Metric | Value |
|---|---:|
| 측정 표본 수 | 136 |
| 평균 | 7.677 ms |
| P50 | 6.856 ms |
| P95 | 11.999 ms |

P95 추론시간 자체는 일반적인 API 처리 관점에서 충분히 작다. 다만 서비스의 공식 latency 허용 기준이 별도로 정의되어 있지 않으므로, 이 문서에서는 절대적인 SLA 통과로 선언하지 않는다.

## 6. v1/v2 공통 Holdout 비교

두 모델에 동일한 leak-free holdout 190건을 입력했다. 각 모델은 artifact에 저장된 threshold를 그대로 사용했다.

| Metric | v1 | v2 | Delta (v2-v1) |
|---|---:|---:|---:|
| Accuracy | 0.5526 | 0.5579 | +0.0053 |
| Precision | 0.5526 | 0.5556 | +0.0029 |
| Recall | 1.0000 | 1.0000 | 0.0000 |
| F1 | 0.7119 | 0.7143 | +0.0024 |
| F2 | 0.8607 | 0.8621 | +0.0014 |
| TN | 0 | 1 | +1 |
| FP | 85 | 84 | -1 |
| FN | 0 | 0 | 0 |
| TP | 105 | 105 | 0 |

표본 단위 비교:

| 결과 | 표본 수 |
|---|---:|
| 두 모델 모두 정답 | 105 |
| 두 모델 모두 오답 | 84 |
| v1만 정답 | 0 |
| v2만 정답 | 1 |

McNemar exact test:

| 항목 | 값 |
|---|---:|
| 불일치 표본 수 | 1 |
| p-value | 1.0000 |
| 유의수준 0.05에서 유의함 | No |

v2는 v1보다 정상 메시지 1건을 추가로 맞혔지만 통계적으로 유의한 개선이 아니다. 두 모델 모두 holdout의 정상 메시지 대부분을 피싱으로 판정하므로 Stacking 단독 운영에 필요한 정상 메시지 판별 성능을 충족하지 못한다.

## 7. Artifact 및 재현성

### 7.1 Artifact 경로와 hash

기존 v1 artifact는 보존했으며 v2는 별도 경로에 저장했다.

| Version | 경로 | Model SHA-256 |
|---|---|---|
| v1 | `data_science/SMSModel/artifacts/stacking/model.joblib` | `fa99576f1d655625410ddb3b6b36ab55faca5a780f296dae9041e37593c063fe` |
| v2 | `data_science/SMSModel/artifacts/stacking/v2/model.joblib` | `b13c6f9df63a2e2229e6db79b5d854ee07621970053b2bc0824d4e80480c48ba` |

- v2 model configuration SHA-256: `1d415c94652cd07c7803ace9e0c8578738dc678f9be62e31ec8ab3cba948b057`
- Stacking artifact에는 base model vectorizer가 함께 직렬화되므로 model hash가 vectorizer 상태도 포함한다.
- 저장 직후 artifact를 다시 로드하고 Test 확률 배열이 저장 전 결과와 허용오차 `1e-12` 안에서 동일한지 검증했다.
- 운영 v1 artifact는 덮어쓰지 않았다.

### 7.2 실행 명령

저장소 루트에서 다음 명령을 사용한다.

```powershell
python -m data_science.SMSModel.run_stacking_training
python -m data_science.SMSModel.run_stacking_holdout_comparison
```

관련 테스트:

```powershell
pytest tests/data_science/SMSModel/test_stacking_training.py -q
pytest tests/data_science/SMSModel/test_stacking_reporting.py -q
pytest tests/data_science/SMSModel/test_stacking_holdout_comparison.py -q
pytest tests/data_science/SMSModel/evaluation/test_metrics.py -q
```

생성 보고서:

- `data_science/SMSModel/reports/stacking_v2/test_evaluation.json`
- `data_science/SMSModel/reports/stacking_v2/test_evaluation.md`
- `data_science/SMSModel/reports/stacking_v2/holdout_comparison_summary.json`
- `data_science/SMSModel/reports/stacking_v2/holdout_comparison_predictions.csv`

## 8. 채택 기준 판정

| 채택 기준 | 결과 | 판정 |
|---|---|:---:|
| 기존 모델 대비 피싱 Recall이 악화되지 않음 | v1 1.0, v2 1.0 | Pass |
| 정상 메시지 FP 감소 | 85 → 84, 1건 감소 | Pass (미미함) |
| FP 감소가 통계적으로 유의함 | McNemar p=1.0 | Fail |
| 치명적인 피싱 미탐 유형이 없음 | 관측 Test 유형의 FN 0건 | Pass |
| 정상 메시지를 서비스 가능한 수준으로 분류 | Test FP 59/60, holdout FP 84/85 | Fail |
| P95 추론시간이 실용적임 | 11.999 ms | Pass |
| 결과 없음·예외·엔진 실패가 없음 | 모두 0건 | Pass |
| 저장·로드 후 예측 동일 | 검증 완료 | Pass |

Recall을 유지했다는 사실만으로는 채택할 수 없다. 낮은 threshold가 거의 모든 메시지를 피싱으로 분류하면서 Recall을 확보했기 때문이다. 이 모델을 단독 운영하면 정상 사용자 메시지 대부분에 경고가 발생할 가능성이 높다.

## 9. 최종 결정 및 후속 권고

**Decision: `REJECT`**

Stacking v2는 Issue #78의 평가 기준상 단독 운영 후보로 채택하지 않는다. v2 artifact는 운영 경로로 승격하지 않고 실험 artifact로만 보존한다.

후속 실험에서는 다음을 권장한다.

1. Validation threshold 선택 조건에 최소 Precision 또는 최대 정상 FPR을 추가한다.
2. `Recall >= 목표` 후보 중 F2만 최대화하는 현재 정책을 비용 기반 목적함수로 변경한다.
3. 일상대화, 정상 금융 알림, 정상 배송 안내와 같이 FP가 집중된 hard-negative 데이터를 Train에 보강한다.
4. 보강 데이터는 향후 holdout과 template/fingerprint가 겹치지 않도록 생성 단계에서 차단한다.
5. 새로운 데이터와 threshold 정책을 사용할 경우 별도 이슈와 새로운 artifact version으로 평가한다.

이번 실험 결과만으로 기존 운영 모델 또는 운영 정책을 변경하지 않는다.
