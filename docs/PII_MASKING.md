# PII 마스킹 토큰 형식

Spring `PiiMaskingService`에서 RabbitMQ 페이로드 생성 직전에 마스킹 후 전달 <br>
FastAPI는 마스킹된 텍스트를 수신하며, 토큰 형식 변경 시 BE 레포와 동시에 수정해야 함

## 토큰 목록

| 토큰 | 의미 |
|---|---|
| `[RRN]` | 주민등록번호 |
| `[CARD]` | 카드번호 |
| `[ACCOUNT]` | 계좌번호 |
| `[PHONE]` | 전화번호 |
| `[EMAIL]` | 이메일 주소 |

## FastAPI 처리 방식

- **NB 분석기** (`naive_bayes_analyzer.py`): 수신된 마스킹 토큰 기준으로 전처리 후 추론
- **규칙 엔진** (`rule_based_analyzer.py`): `[ACCOUNT]`, `[CARD]` 토큰 OR 패턴 조건으로 점수 부여
- **Gemini**: 마스킹 텍스트 그대로 분석 (문맥 이해 가능)

## 변경 절차

1. `SafeFam_BE`와 `SafeFam_AI` 양쪽 레포에서 토큰 형식 동시 수정
2. 양쪽 테스트 통과 확인
3. Spring → FastAPI 순서로 배포
