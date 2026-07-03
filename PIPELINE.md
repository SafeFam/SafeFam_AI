# SafeFam 피싱 탐지 모델 파이프라인

SMS(문자)와 Voice(보이스피싱 통화 STT), 두 개의 독립된 나이브베이즈 분류기.
구조는 동일(전처리 → 구조적 피처 → 벡터화 → train/val/test → 튜닝 → 평가)하고
도메인에 맞게 피처와 벡터화 방식만 다르게 설정.

| | SMS | Voice |
|---|---|---|
| 노트북 | `SMSModel/SMSData.ipynb` | `VoiceModel/VoiceData.ipynb` |
| 원본 데이터 | `Data/SMSData/phishing_total_dataset_2705.csv` | `Data/CallData/metadata_clean.csv` |
| 입력 | 문자 메시지 원문 | STT 변환 통화 텍스트(`text_clean`) |
| 아티팩트 | `phishing_model_artifact.pkl`, `phishing_vectorizer.pkl` | `voice_model_artifact.pkl`, `voice_vectorizer.pkl` |

---

## 파이프라인 흐름

**1. 전처리**
- SMS: URL/전화번호/숫자/금액을 `<URL>` 등 토큰으로 정규화 → 정규화 텍스트 기준 중복 제거 (2,705 → 811건)
- Voice: `metadata_clean.csv`에 STT 정제 텍스트(`text_clean`)가 이미 준비되어 있어 그대로 사용

**2. 구조적 피처** (규칙 기반 boolean, n-gram이 못 잡는 신호 보강)
- SMS(6개): `has_url`, `has_short_url`, `has_phone`, `has_amount`, `has_web_tag`, `is_long_text`
- Voice(8개): `has_urgency`, `has_authority`, `has_money`, `has_personal`, `has_threat`, `has_safe_acct`, `has_install`, `is_long_text`

**3. 벡터화**: `CountVectorizer(analyzer="char_wb", ngram_range=(2,4))` — 문자 n-gram이라 한국어 조사 변형(계좌/계좌는/계좌를)에 강건. Voice는 word n-gram도 후보로 비교 후 char_ngram 채택. **fit은 train에만**, val/test는 transform만.

**4. Train / Val / Test 분할** — `parent_call_id`(통화 단위) 또는 정규화 텍스트 기준으로 **그룹 단위 분할**, val은 튜닝 전용·test는 최종 평가 1회 전용.

**5. 모델**: `ComplementNB` + `CalibratedClassifierCV(isotonic)` 확률 보정. `alpha × threshold` 격자 탐색을 **validation set**으로만 수행 (`Recall(phishing) ≥ 0.85` 우선, 그 안에서 `Recall(normal)` 최대).

**6. 최종 평가**: 튜닝에 한 번도 안 쓴 test set으로 1회만 평가.

**7. 해석**: `feature_log_prob_` 차이로 피싱/정상 근거 키워드 top-20 시각화(`feature_importance.png`) + 근거 키워드 군집화(`feature_clusters.png`) + 전체 피처 점수 CSV(`feature_scores_full.csv`).

---

## 최종 결과 (가장 최근 학습 기준)

| | Train | Val | Test | Accuracy | Recall(phishing) | Recall(normal) |
|---|---|---|---|---|---|---|
| **SMS** | 567 | 122 | 122 | 0.93 | 0.9242 | 0.9464 |
| **Voice** | 1,199 | 257 | 257 | 1.00 | 0.9912 | 1.0000 |

- SMS: `alpha=0.1, threshold=0.7`
- Voice: `alpha=0.01, threshold=0.55` (char n-gram)

---

## 누수(leakage) 방지 포인트

- 벡터라이저 fit은 항상 train에만.
- SMS: 정규화 텍스트 기준 중복 제거를 split 이전에 수행. test를 튜닝에 재사용하던 과거 버그를 val 분리로 수정.
- Voice: 최근 정상(normal) 데이터 300건을 증강 추가했다가 **같은 통화의 증강 변형이 train/val/test에 흩어지는 그룹 누수**가 발생 → `parent_call_id` + 완전 동일 텍스트를 하나로 묶는 Union-Find 그룹 기준으로 재분할해 해결. 재분할 후 노트북 내장 검증 셀에서 통화/텍스트 중복 0건 확인.

---

## 정리 이력

- `SMSModel/phishing_nb_model.pkl` (보정 없는 구버전 산출물) 삭제
- Voice 검증 셀 미사용 `import shap` 제거, SMS 시각화 셀 상대경로 오타 수정
- `Data/CallData/metadata_clean.csv.bak_before_regroup_split` (재분할 전 백업) 확인 후 삭제
- 미참조 상태로 남아있는 것: `Data/CallData/README.md`, `validation_report.json`, `build_summary.json`,
  `transcription_errors.jsonl`, `metadata.jsonl`, `metadata_clean.jsonl` — Stage 0 EDA 산출물로 추정, 삭제 여부 미결정
