# 피싱·정상 균형 음성 데이터셋

## 구성

- 총 3,000개 WAV
- 피싱 1,500개 / 정상 1,500개
- train 2,100개 / validation 450개 / test 450개
- 각 클래스별 train 1,050개 / validation 225개 / test 225개
- 길이 10~30초
- WAV 16 kHz, mono, 16-bit PCM
- 증강 데이터 없음

## 출처

- 피싱: 기존 금융감독원(FSS) 피싱 통화 원본 중 자동 QA `PASS` 구간
- 정상: AI Hub `상담 음성`(dataSetSn=100) D60 Validation WAV 및 제공 전사
- 개인 통화음원은 포함하지 않았음
- AI Hub 정상 음성은 재전사하지 않고 제공 TXT 라벨을 사용함

AI Hub 원천 데이터의 이용·공유에는 AI Hub 이용정책과 승인 조건이 적용됩니다.

## 분할 및 품질 규칙

- `parent_call_id` 기준으로 원본 통화 단위 분할
- 동일 원본 통화가 train/validation/test에 동시에 들어가지 않음
- 피싱 `REVIEW` 152건 제외
- 30초 초과 전사 구간 및 짧은 응답 4회 이상 연속 구간 추가 제외
- 정상 통화의 연속 발화를 10~30초가 되도록 결합
- validation/test는 원본 기반 데이터만 사용

## 파일

- `audio/phishing/`: 피싱 WAV
- `audio/normal/`: 정상 WAV
- `metadata.csv`: UTF-8 BOM CSV 메타데이터
- `metadata.jsonl`: 동일 내용을 담은 JSON Lines
- `transcription_errors.jsonl`: 제외·해결 불가 원본 기록
- `build_summary.json`: 생성 통계
- `validation_report.json`: 최종 자동 검수 결과

`validation_report.json`의 `passed` 값이 `true`인지 확인한 후 사용하십시오.
