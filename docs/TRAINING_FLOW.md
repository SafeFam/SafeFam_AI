# SMS / Voice 모델 학습 흐름

`PIPELINE.md`가 두 모델의 구조를 표로 요약한 개요 문서라면, 이 문서는 **실제로 어떤 스크립트를 어떤 순서로 실행하면 학습이 되는지**를 단계별로 따라간다.

```
SMSModel/
 └─ train_sms.py                 (단일 스크립트로 완결)

VoiceModel/
 └─ preprocess_voice.py  ─▶  generate_voice_data.py  ─▶  train_voice.py
    (STT 정제)              (정상 통화 증강 합성)         (학습/평가/저장)
```

---

## 0. 사용 기술 스택

두 모델 모두 딥러닝이 아닌 **고전 확률 모델(나이브베이즈) + 규칙 기반 피처**로 구성되어 있다. 실시간성/설명가능성이 필요한 1차 필터링 단계에 맞춰 가볍고 해석 가능한 조합을 택한 것.

| 분류 | 기술 | 사용처 |
|---|---|---|
| 분류 모델 | `sklearn.naive_bayes.ComplementNB` | SMS 전용 / Voice 후보1 — 클래스 불균형에 강한 나이브베이즈 변형 |
| | `sklearn.naive_bayes.MultinomialNB` (+ `class_prior` 수동 보정) | Voice 후보2 — ComplementNB와 격자 탐색으로 비교 후 채택 여부 결정 |
| 확률 보정 | `sklearn.calibration.CalibratedClassifierCV(method="isotonic", cv=5)` | 두 모델 공통 — NB 특유의 0/1 극단 확률을 실제 비율에 맞게 보정해 `risk_score`를 0~100 연속값으로 사용 가능하게 함 |
| 텍스트 벡터화 | `sklearn.feature_extraction.text.CountVectorizer` | `analyzer="char_wb", ngram_range=(2,4)` (SMS 고정 / Voice 후보1) — 형태소 분석기 없이 문자 n-gram만으로 한글 조사 변형 대응 |
| | 동일 클래스, `analyzer="word", ngram_range=(1,2)` | Voice 후보2 — 구어체 STT에 word n-gram이 더 나을 가능성을 검증하기 위한 비교 대상 |
| 피처 엔지니어링 | Python `re` 정규식 기반 규칙 피처 | SMS 6종(URL/단축URL/전화번호/금액/통신사태그/장문), Voice 8종(긴급성/기관사칭/금전/개인정보/위협/안전계좌/앱설치/장문) — n-gram이 못 잡는 구조적 신호 보강 |
| 데이터 분할 | `sklearn.model_selection.StratifiedShuffleSplit` (2단계) | 두 모델 공통 — label 비율 유지하며 train/val/test 분리 |
| | 자체 구현 Union-Find | Voice 전용 — `parent_call_id` + `text_clean` 그룹을 묶어 leak-free 분할을 구조적으로 보장 |
| 하이퍼파라미터 탐색 | 격자 탐색(grid search) 직접 구현 (`for` 이중/삼중 루프) | 두 모델 공통 — `alpha × threshold`(SMS), `벡터화 × 모델 × alpha × threshold`(Voice), `sklearn.model_selection.GridSearchCV` 대신 val 지표(Recall 우선순위) 커스텀 로직으로 직접 순회 |
| 평가 지표 | `sklearn.metrics.classification_report`, `confusion_matrix` | 두 모델 공통 — Recall(phishing) 우선, 그 안에서 Recall(normal) 최대화 |
| 피처 결합 | `scipy.sparse.hstack`, `csr_matrix` | 두 모델 공통 — 텍스트 sparse 벡터 + 구조 피처(dense)를 하나의 sparse 행렬로 결합 |
| 데이터 처리 | `pandas`, `numpy` | 두 모델 공통 — 데이터프레임 조작, 배열 연산 |
| 직렬화 | `joblib` | 두 모델 공통 — `{model, threshold, classes, ...}` 딕셔너리 통째로 pkl 저장/로드 (FastAPI 서빙용) |
| STT 텍스트 정제 | Python `re` 기반 정규식 치환 (`clean_text`) | Voice 전용 — 화자기호/추임새/발음교정/불명확발음 등 7종 패턴 제거, 학습·서빙 코드 공유 |
| 정상 데이터 증강 | 템플릿 기반 규칙적 텍스트 생성 (`random.choice` 조합) | Voice 전용 — 별도 생성 모델 없이 문자열 템플릿 조합으로 정상 통화 850건 합성 |

**의도적으로 사용하지 않은 것**: 딥러닝(RNN/BERT 등), 형태소 분석기(Mecab/Okt 등), 외부 임베딩 — char n-gram만으로 충분한 성능(SMS 정확도 0.93, Voice 1.00)을 확인했고, 가벼운 추론 속도와 `feature_log_prob_` 기반 키워드 해석가능성을 우선했기 때문(`PIPELINE.md`의 "해석" 단계 참고).

---

## 1. SMS 모델 — `SMSModel/train_sms.py`

한 파일 안에서 로드부터 저장까지 전부 처리되는 단일 스크립트. `python train_sms.py` 실행 시 `main()`이 아래 순서로 호출된다.

```
① load_data()          Data/SMSData/phishing_total_dataset_2705.csv 로드
② split_data()          Stratified 2단계 split
③ _extract_struct_features()  구조 피처 6개 추출
④ build_vectorizer() + build_feature_matrix()   벡터화 + 피처 결합
⑤ train_and_tune()      alpha × threshold 격자 탐색 (val 기준)
⑥ verify_probability_distribution()   보정 확률 분포 점검
⑦ evaluate()            test set 최종 평가 (1회)
⑧ save_artifacts()      pkl 2종 저장
```

### ① 데이터 로드 + 정규화 + 중복 제거 — `load_data()`
- 원본 2,705건 로드, 필수 컬럼(`text`, `label`, `type`, `has_url`) 및 결측치 검증
- `_normalize_text()`로 URL → `<URL>`, 전화번호 → `<전화번호>`, 6자리+ 숫자 → `<긴숫자>`, 금액 → `<금액>` 치환
- **정규화된 텍스트(`text_norm`) 기준 중복 제거**: URL/금액만 다른 동일 템플릿을 하나로 취급해야 train/test 누수를 막을 수 있음 → **2,705건 → 811건**

### ② Train / Val / Test 분리 — `split_data()`
- `StratifiedShuffleSplit`을 2단계로 적용: 전체 → (train+val) / **test(15%)**, 다시 (train+val) → **train** / **val(15%)**
- val = 하이퍼파라미터 튜닝 전용, test = 최종 평가 1회 전용 (튜닝에 재사용 금지)

### ③ 구조적 피처 추출 — `_extract_struct_features()`
n-gram이 못 잡는 신호를 boolean 6개로 보강: `has_url`, `has_short_url`(단축URL, phishing 27.8% vs normal 0.4%로 최강 단독 신호), `has_phone`, `has_amount`, `has_web_tag`(`[Web발신]` 등 — 정상 신호 방향), `is_long_text`(100자 초과)

### ④ 벡터화 + 피처 결합 — `build_vectorizer()` / `build_feature_matrix()`
- `CountVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=2, max_df=0.95, max_features=8000)` — 문자 n-gram이라 조사 변형(계좌/계좌는/계좌를)에 강건
- 텍스트 피처(sparse) + 구조 피처(6열) 를 `hstack`으로 결합
- **fit은 train에만**(`fit=True`), val/test는 `transform`만 — 누수 방지

### ⑤ 격자 탐색 — `train_and_tune()`
- `ComplementNB(alpha)`를 `CalibratedClassifierCV(method="isotonic", cv=5)`로 감싸 확률 보정
  → 원 확률이 0/1 극단으로 몰리는 문제 해결, `risk_score`가 0~100에 고르게 분산되도록
- `alpha ∈ {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0}` × `threshold ∈ [0.30, 0.75] (0.05 간격)` 전수 탐색, **val set**으로만 평가
- 선택 기준: `Recall(phishing) ≥ 0.85`를 만족하는 조합 중 `Recall(normal)` 최대 (미달 시 `Recall(phishing)` 최대로 폴백)

### ⑥ 확률 분포 검증 — `verify_probability_distribution()`
- 보정 후 `prob_phishing`이 LOW(<0.40) / MEDIUM(0.40~0.70) / HIGH(≥0.70) 구간에 고르게 분포하는지 확인
- MEDIUM 구간이 실제 서비스에서 Claude API 에스컬레이션 대상이므로, 이 구간에 샘플이 없으면 경고 출력

### ⑦ 최종 평가 — `evaluate()`
- 튜닝에 한 번도 쓰지 않은 **test set**으로 최적 threshold 적용 후 1회 평가 (classification report + confusion matrix)

### ⑧ 아티팩트 저장 — `save_artifacts()`
- `phishing_model_artifact.pkl` = `{model, threshold, classes}` (FastAPI 로드용)
- `phishing_vectorizer.pkl` = 학습된 `CountVectorizer`

### Stacking + Gemini 하이브리드 정책 선정

- Stacking validation 확률과 Gemini validation 점수를 fingerprint 순서로
  정렬해 동일 샘플끼리 비교한다.
- 목표 Recall을 만족하는 후보 중 F2가 높은 정책을 우선하고, F2가 같으면
  Gemini 호출률이 낮은 정상/피싱 경계값을 선택한다.
- Gemini 결과는 fingerprint 기반 JSON 캐시에 저장하여 quota 제한이나 중단
  이후에도 성공한 호출을 재사용한다.
- 선택된 Recall, F2, 호출률과 경계값은 policy report 및 Stacking metadata의
  `hybrid_policy`에 저장한다.
- 현재 캐시는 123건 중 17건만 기록됐고 사용 가능한 결과는 13건이다.
  unavailable 4건과 누락 항목이 남아 있어 최종 임계값과 호출률은
  provisional이며, 이 상태에서 정책 선정이 중단되는 것은 의도된 동작이다.

### 서빙 시 참고 — `predict_risk_score()`
학습 스크립트에 함께 정의된 추론 함수. 원문 텍스트 → 동일한 정규화/구조피처/벡터화 파이프라인 통과 → 보정된 `prob_phishing × 100`을 `risk_score`로, `RISK_HIGH_THRESHOLD=70` / `RISK_MEDIUM_THRESHOLD=40` 기준으로 `risk_level`(HIGH/MEDIUM/LOW) 산출.

---

## 2. Voice 모델 — 3단계 스크립트

Voice는 데이터 준비가 무거워서 3개 스크립트로 분리되어 있다. **반드시 아래 순서대로 실행**해야 한다.

```
1) python preprocess_voice.py    → Data/CallData/metadata_clean.csv  (원본 STT 정제)
2) python generate_voice_data.py → metadata_clean.csv 덮어씀          (정상 통화 합성 데이터 추가)
3) python train_voice.py         → 학습 / 평가 / 아티팩트 저장
```

### 1) `preprocess_voice.py` — STT 텍스트 정제
- `Data/CallData/metadata.csv`(수동 전사 원본) 로드
- `clean_text()`로 화자기호(`n/`, `b/`), 추임새(`아/`, `음/`), 발음교정 표기 `(A)/(B)`→A, 불명확 발음(`*`), 휴지기호(`+`), 말끊김(`..`/`...`), STT 오인식(`xx`/`XX`) 제거
  - 이 처리들은 **phishing 전사에서만 등장**(예: 말끊김 phishing 53건 vs normal 0건)해 라벨과 무관한 "전사 스타일 차이"를 모델이 오학습하지 않도록 하기 위함
- 결과를 `text_clean` 컬럼으로 추가, 잔존 기호 0건 및 평균 길이 90% 이상 유지를 `_validate()`로 어설션 검증 후 `metadata_clean.csv` 저장
- **동일 `clean_text()`를 FastAPI 실시간 추론에서도 import** — 학습/서빙 전처리 불일치 방지가 목적

### 2) `generate_voice_data.py` — 정상 통화 합성 데이터 증강
- 1)의 결과물 `metadata_clean.csv`를 읽어 **정상(normal) 통화 합성 데이터를 추가**하고 같은 파일에 덮어쓴다
- 유형별 생성: 병원/약국(50), 배달/택배(50), 통신사 CS(50), 은행 정식 안내(50), 지인/가족 일상(200, v2에서 50→200 증량), 쇼핑몰 CS(50), 금융기관 진짜 안내(150), 수사기관 진짜 안내(150), 쌍방향 일상 대화(100, v2 신규) — 총 850건
- 설계 원칙: `is_augmented=True` / `source_dataset="synthetic"`로 식별 가능하게 태깅, 고정 면책 문구 미사용(v1에서 과적합 원인으로 확인됨), 마무리 표현 10종을 무작위 배분해 특정 표현 과의존 방지
- `parent_call_id` prefix(`D63_H01` 등)로 유형을 구분한다. 생성 단계의
  train/val/test 표시는 참고용이며, 최종 학습에서는 이 값을 사용하지 않고
  `_leak_free_split()`이 그룹 단위로 다시 분할한다.

### 3) `train_voice.py` — 학습 / 튜닝 / 평가
```
① load_data() → _leak_free_split()   그룹 단위 leak-free 3분할
② _extract_struct_features()          STT 특화 구조 피처 8개
③ compare_vectorizers()               char n-gram vs word n-gram 비교
   └─ train_and_tune()                 (ComplementNB / MultinomialNB) × alpha × threshold 격자 탐색
④ evaluate()                          test set 최종 평가
⑤ save_artifacts()                    pkl 2종 저장 (+ vectorizer_type)
```

#### ① 데이터 로드 + **leak-free 분할** — `load_data()` / `_leak_free_split()`
과거 메타데이터의 `split` 컬럼을 그대로 썼을 때 두 종류의 누수가 발견됨:
1. 증강 600건이 `parent_call_id`로 그룹화되지 않고 행 단위로 개별 split 배정 → 같은 통화의 세그먼트가 train/test에 흩어짐
2. 보이스피싱 시나리오가 스크립트 템플릿을 재사용해, 서로 다른 통화인데 `text_clean`이 완전히 동일한 경우 존재 → `parent_call_id`만 묶어도 텍스트 누수가 남음

→ `parent_call_id`와 `text_clean` **두 키 중 하나라도 같으면 같은 그룹으로 묶는 Union-Find**를 직접 구현해 메타데이터 split을 무시하고 그룹 단위 `StratifiedShuffleSplit` 2단계로 재분할한다. 먼저 전체 그룹의 15%를 test로 분리하고, 남은 85%에서 `15/85` 비율을 validation으로 분리하므로 최종 validation도 원본 전체의 15%다. 결과 비율은 train/validation/test 약 70/15/15이며, 재분할 후 통화/텍스트 중복 0건을 확인한다.

#### ② 구조적 피처 — `_extract_struct_features()` (STT 특화 8개)
`has_urgency`(긴급성 유도: "지금 바로", "즉시"), `has_authority`(검찰/경찰/금감원 등 사칭), `has_money`(대출/이자/송금 등), `has_personal`(주민번호/계좌번호 등 요구), `has_threat`(체포/구속/기소 등 위협), `has_safe_acct`(안전계좌 유도), `has_install`(앱설치/원격제어), `is_long_text`(200자 초과)

#### ③ 벡터화 비교 + 격자 탐색 — `compare_vectorizers()` / `train_and_tune()`
- 후보 2종을 **각각** train에 fit → val로 평가: `char_wb(2,4)`(SMS와 동일) vs `word(1,2)`(구어체에 더 직관적일 수 있어 후보로 추가)
- 각 후보마다 `ComplementNB` / `MultinomialNB`(class_prior로 정상:피싱 비율 보정) × `alpha` 8종 × `threshold` 9종을 전수 탐색, 모두 `CalibratedClassifierCV(isotonic, cv=5)`로 확률 보정
- 선택 기준은 SMS와 동일(`Recall(phishing) ≥ 0.85` 하에 `Recall(normal)` 최대) — 벡터화 방식 자체도 이 기준으로 최종 채택

#### ④ 최종 평가 — `evaluate()`
- val로 고른 벡터화/모델/alpha/threshold를 **test set**(튜닝에 전혀 쓰이지 않은 그룹)에 1회만 적용해 classification report + confusion matrix 출력

#### ⑤ 아티팩트 저장 — `save_artifacts()`
- `voice_model_artifact.pkl` = `{model, threshold, classes, vectorizer_type}`
- `voice_vectorizer.pkl` = 채택된 벡터라이저
- SMS와 파일명을 분리해 혼용 방지

### 서빙 시 참고 — `predict_risk_score()`
`preprocess_voice.py`로 이미 정제된 `text_clean` 입력 → 구조피처 8개 + 벡터화 → `prob_phishing × 100`을 `risk_score`로, SMS와 동일한 HIGH(≥70)/MEDIUM(40~69)/LOW(<40) 기준 적용. 응답에 `"modality": "voice"` 포함해 SMS 응답과 구분.

---

## 3. 두 모델 학습 흐름 비교

| 단계 | SMS | Voice |
|---|---|---|
| 실행 스크립트 수 | 1개 (`train_sms.py`) | 3개 (`preprocess_voice.py` → `generate_voice_data.py` → `train_voice.py`) |
| 전처리 | 정규식 토큰화 + 정규화 텍스트 중복 제거 | STT 전사 기호 정제(`clean_text`) — 학습/서빙 공유 |
| 데이터 증강 | 없음 | 정상 통화 합성 850건 추가 (`generate_voice_data.py`) |
| 분할 단위 | 정규화 텍스트 중복 제거 후 개별 행 | `parent_call_id` + `text_clean` Union-Find 그룹 |
| 구조 피처 | 6개 (URL/전화번호/금액 등 텍스트 신호) | 8개 (긴급성/기관사칭/위협 등 구어체 신호) |
| 벡터화 탐색 | char n-gram 1종 고정 | char n-gram vs word n-gram 비교 후 채택 |
| 모델 후보 | `ComplementNB` 1종 | `ComplementNB` vs `MultinomialNB`(class_prior 보정) 비교 |
| 확률 보정 | `CalibratedClassifierCV(isotonic, cv=5)` | 동일 |
| 튜닝/평가 분리 | val(튜닝) / test(평가 1회) 동일 원칙 | 동일 |
| 산출 아티팩트 | `phishing_model_artifact.pkl`, `phishing_vectorizer.pkl` | `voice_model_artifact.pkl`, `voice_vectorizer.pkl` |

공통 원칙: **벡터라이저 fit은 항상 train에만**, **val로 튜닝 / test로 단 1회 평가**, **risk_score(0~100) → HIGH/MEDIUM/LOW 3단계**, **MEDIUM 구간은 Claude API 에스컬레이션 대상**.

전체 파이프라인 상 위치, 최종 성능 지표, 정리 이력은 `PIPELINE.md` 참고.
