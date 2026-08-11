# 스미싱 탐지 파이프라인 개편 정리 (Issue #19 및 후속 스코어링 개선)

## 개요

기존에는 모든 문자에 대해 Gemini API를 호출하는 단일 트랙 구조였다. 이번 작업으로 다음 구조로 전환했다.

```
문자 텍스트
  ├─ 텍스트 트랙: 나이브 베이즈(1차, 로컬/무료) → SAFE 미만이면 Gemini(2차) 에스컬레이션
  ├─ URL 트랙: 단축 URL 추적 → Google Safe Browsing(1차) → VirusTotal(2차 백업)
  └─ 규칙 트랙: 금융기관 DB / 금융 키워드 / 계좌·카드번호 패턴 / 로컬 도메인 룰
        │
        ▼
  3중 가중치 스코어링 (URL 유무에 따라 배점 재조정)
        │
        ▼
  최종 점수(0~100) + 등급(LOW/MEDIUM/HIGH)
```

---

## 1. 나이브 베이즈 모델 연동

**`app/service/security/naive_bayes_text_analyzer.py`** (신규)

- 기존에 학습돼 저장소에 커밋되어 있던 `data_science/SMSModel/artifacts/phishing_model_artifact.pkl`(CalibratedClassifierCV + ComplementNB), `phishing_vectorizer.pkl`을 로드
- 전처리(URL/전화번호/금액 마스킹, 6개 구조적 피처)는 학습 스크립트(`train_sms.py`)와 동일하게 재구현 — 학습/서빙 피처 불일치 방지
- 모델 로드 실패 시 `UNKNOWN` 등급 + 에러 메시지로 fail-safe 처리 (SAFE로 오판하지 않음)
- `requirements.txt`에 `scikit-learn`, `scipy`, `numpy`, `joblib` 추가

**알려진 한계**: 학습 데이터에 "엄마"라는 단어가 포함된 정상 문자가 0건이라(전부 자녀 사칭 스미싱 템플릿), 나이브 베이즈 단독으로는 "엄마 오늘 저녁 메뉴 뭐야?" 같은 문장을 `DANGEROUS(97점)`로 오판한다. 실전에서는 아래 2번 스위칭 로직 덕분에 Gemini가 재검증해서 최종 오탐으로 이어지지는 않지만, 모델 자체의 재학습 여지는 남아있다.

## 2. 1차 나이브베이즈 + 2차 Gemini 스위칭 로직

**`app/service/scan_service.py`** — `_analyze_text_hybrid()`

- 나이브 베이즈가 정상 로드되고 `SAFE`로 판정한 경우에만 Gemini 호출 스킵
- 그 외(의심/위험 판정, 또는 모델 로드 실패)는 Gemini 2차 검증 실행, 1차 점수를 `stage1_naive_bayes`로 응답에 동봉
- Gemini 호출이 실패(rate limit 등)해도 이미 의심 판정한 나이브 베이즈 점수를 0으로 깎지 않고 그대로 신뢰 (fail-safe)

## 3. URL 리다이렉트 추적기 보완

**`app/utils/url_tracker.py`**

- **상대경로 리다이렉트 버그 수정**: `Location` 헤더가 `/`로 시작하는 경우만 처리하던 것을, 절대 URL이 아닌 모든 경우(프로토콜 상대경로 `//`, 슬래시 없는 상대경로 등)로 확장
- **SSRF 방어 추가**: 매 리다이렉트 홉마다 호스트를 DNS로 조회해 사설 대역/루프백/링크로컬/클라우드 메타데이터 주소(`169.254.169.254` 등)면 요청 자체를 차단. DNS 조회 실패도 안전하게 차단(fail-closed)

## 4. 3중 스코어링 엔진 — 나이브 베이즈 + Gemini 하이브리드 결합

**`app/utils/scoring_engine.py`** — `_combine_text_track_score()`

- 나이브 베이즈 SAFE 판정으로 Gemini 스킵: 나이브 베이즈 점수 그대로 사용
- Gemini 호출 실패: 나이브 베이즈 점수를 그대로 신뢰 (API 장애로 위험 점수가 0으로 깎이는 것 방지)
- 둘 다 정상 수행: `0.3 × 나이브베이즈 + 0.7 × Gemini` 가중 평균

## 5. Swagger 기반 E2E 통합 테스트

**`tests/e2e/test_analyze_endpoint.py`** (신규)

- `fastapi.testclient.TestClient`로 실제 `POST /api/analyze` 엔드포인트를 통해 전체 파이프라인 검증
- Gemini/GSB+VT(유료·외부 API)만 Mock으로 우회, 나이브 베이즈·URL 추적기는 실제 로직 사용
- 시나리오: 일상 대화(Gemini 스킵), URL 없는 스미싱(에스컬레이션), URL 포함 문자, 응답 스키마 계약, 필수 필드 누락 422, OpenAPI 노출 확인

## 6. GSB/VT/도메인룰 확정 악성 하드 오버라이드

**`app/service/security/hybrid_url_engine.py`**, **`app/utils/scoring_engine.py`**

- **GSB 블랙리스트 실제 등재 확인** → 무조건 HIGH (텍스트 문맥과 무관)
- **VT 다수 엔진 합의**(5개 이상 동시 탐지) → 마찬가지로 HIGH. 단, 3~4개 소수 탐지는 오탐 벤더 가능성을 고려해 기존처럼 가중치 점수로만 반영
- **로컬 도메인 룰**(`.ru`, `testsafebrowsing` 등) 매치 → HIGH
- 등급-점수 표기 일관성을 위해 오버라이드 시 `final_score`도 HIGH 임계치(70점) 이상으로 보정 (이미 더 높으면 유지)

## 7. URL 없을 때 트랙 가중치 재분배 (65:35)

**`app/utils/scoring_engine.py`**

- URL 있음: LLM 50% + URL 30% + 규칙 20% (기존과 동일)
- URL 없음: URL 트랙(30%)이 성립하지 않으므로 LLM 65% + 규칙 35%로 재분배
- **효과**: URL 없는 순수 텍스트형 스미싱이 LLM 트랙 상한(50점)에 막혀 최대 MEDIUM에 갇히던 구조적 결함 해소
- `app/dto/schemas.py`의 `ContributionBreakdown` 필드 상한도 `llm 0~65`, `rules 0~35`로 확장

## 8. 규칙 기반 트랙 신규 구축

**`app/service/security/rule_based_analyzer.py`** (신규)

로컬 결정론적 규칙 4종을 0~100점으로 합산:

| 신호 | 배점 | 비고 |
|---|---|---|
| 금융기관/공공기관 명칭 언급 | +15 | 은행/카드사/증권/보험사/검찰청 등 50여 개 실명 DB 대조. 단순 언급만으론 약한 신호로 설계 |
| 금융 긴급 키워드 | 카테고리당 +10, 최대 +30 | 이체/대출/계좌정지/명의도용/카드정지/압류연체/긴급확인 등 7개 카테고리 |
| 계좌번호 패턴 | +30 | 하이픈 구분 또는 10~14자리 연속 숫자 |
| 카드번호 패턴 | +30 | 4-4-4-4 형식 또는 16자리 연속 |
| 로컬 도메인 룰(.ru 등) | 100 (단독 만점) | 기존 로직 이관 |

- 응답에 `rule_analysis` 필드 추가 — 어떤 규칙이 왜 매치됐는지 투명하게 노출

## 9. VirusTotal 점수를 "탐지 엔진 비율" 기반으로 재계산

**`app/service/security/virustotal.py`**

- 기존: `malicious × 0.15 + suspicious × 0.05` (임의 개수 기반 공식)
- 변경: `malicious/전체엔진수 + (suspicious/전체엔진수) × 0.5` (실제 비율 기반)
- 위 공식의 `raw_score` 단위는 **0~1**이다. RabbitMQ 외부 이벤트에서는
  `app/infrastructure/rabbitmq/result_factory.py`의 `_url_score()`가 한 번만
  0~100 점수로 변환한다. 호출자는 변환된 값을 다시 100배 하면 안 된다.
- 경계 예시: `raw_score=0`은 외부 점수 `0`, `raw_score=1.0`은 외부 점수
  `100`이다. 전체 엔진 수가 0이면 raw score와 외부 점수 모두 `0`이다.
- `is_malicious` 판정(엔진 3개 이상 등 절대 개수 기준)은 그대로 유지 — 비율과는 별개 기준
- 전체 엔진 풀이 크면(70~90개) 비율만으로는 점수가 낮아 보일 수 있는데, `hybrid_url_engine.py`에 이미 있던 개수 기반 하한선 로직(`max(비율점수, 하한선)`)이 이를 보완

---

## 최종 스코어링 요약

```text
URL 있음:  최종점수 = 선택된 텍스트 트랙(50%) + URL 트랙(30%) + 규칙 트랙(20%)
URL 없음:  최종점수 = 선택된 텍스트 트랙(65%)                  + 규칙 트랙(35%)

단, 아래 중 하나라도 해당하면 위 계산과 무관하게 HIGH 강제:
  - GSB 블랙리스트 등재 확인
  - VT 5개 이상 엔진 동시 탐지
  - 로컬 도메인 룰(.ru 등) 매치

등급: 0~39 LOW / 40~69 MEDIUM / 70~100 HIGH
```

### 최종 등급과 Gemini 라우팅 임계값

- `0~39`, `40~69`, `70~100`은 최종 합산 점수를 LOW/MEDIUM/HIGH로
  분류하는 등급 임계값이다.
- `STACKING_NORMAL_PROBABILITY_MAX`와
  `STACKING_PHISHING_PROBABILITY_MIN`은 Stacking 확률이 불확실한지를
  판단하여 Gemini를 호출하는 별도 라우팅 임계값이다.
- 현재 런타임 기본값은 각각 `0.1`, `0.9`인 보수적 임시값이다. Gemini
  validation 수집 완료 후 Recall, F2와 호출률을 기준으로 교체해야 한다.

### Gemini validation 상태

- validation 전체 123건 중 캐시에 기록된 항목은 17건이며, 사용 가능한
  결과는 13건이다.
- Gemini quota 제한으로 4건이 unavailable 상태이고 나머지 항목도 아직
  수집되지 않았다.
- fingerprint 캐시는 성공한 호출을 재사용하기 위한 재개 지점이다.
  unavailable 또는 누락 항목이 남아 있는 동안 임계값 선정이 실패하는 것은
  의도된 동작이며 회귀가 아니다.
- 최종 Recall, F2, Gemini 호출률과 policy metadata는 전체 결과 수집 후
  확정한다.

## 테스트 현황

기존 변경 범위 테스트 59개와 함께 최신 로컬 전체 테스트를 실행했다.
전체 결과는 **459 passed, 3 deselected**였으며, 아래 목록은 과거 변경
범위를 설명하기 위한 참고 목록이다.

- `tests/security/test_naive_bayes_text_analyzer.py`
- `tests/security/test_rule_based_analyzer.py`
- `tests/security/test_hybrid_url_engine.py`
- `tests/security/test_virustotal.py`
- `tests/security/test_scoring_engine.py`
- `tests/service/test_scan_service.py`
- `tests/url/test_url_tracer.py`
- `tests/e2e/test_analyze_endpoint.py`

Gemini 실호출 기반 최종 정책 검증은 quota 제한으로 완료하지 못했다.
따라서 최종 라우팅 임계값과 예상 Gemini 호출률은 아직 provisional이다.
