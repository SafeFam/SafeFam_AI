# SMS Dataset Split Summary

## Dataset

- Schema version: `1`
- Dataset fingerprint: `c353f1af8ebda78dd739c9129ec3c5e0cea4f5191641abc46b54b9c933395bae`
- Total rows: 859
- Template groups: 578

## Configuration

- Template similarity threshold: `0.88`
- Template n-gram range: `[2, 5]`
- Random state: `42`

## Split Overview

| Split | Rows | Ratio | Groups | Normal | Phishing |
|---|---:|---:|---:|---:|---:|
| train | 604 | 70.31% | 388 | 281 (46.52%) | 323 (53.48%) |
| validation | 129 | 15.02% | 98 | 71 (55.04%) | 58 (44.96%) |
| test | 126 | 14.67% | 92 | 51 (40.48%) | 75 (59.52%) |

## Leakage Validation

- Passed: `True`
- Template group overlap count: `0`
- Fingerprint overlap count: `0`

## Message Type Distribution

### Train

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 41 | 6.79% |
| 금융기관사칭 | 73 | 12.09% |
| 기타피싱 | 43 | 7.12% |
| 대출_사기 | 2 | 0.33% |
| 이벤트당첨사칭 | 62 | 10.26% |
| 일상대화 | 105 | 17.38% |
| 정부공공기관사칭 | 17 | 2.81% |
| 정상알림톡 | 165 | 27.32% |
| 정상포인트소멸알림 | 11 | 1.82% |
| 중고거래_사기 | 5 | 0.83% |
| 지인사칭 | 25 | 4.14% |
| 택배사칭 | 34 | 5.63% |
| 투자_리딩방 | 21 | 3.48% |

### Validation

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 14 | 10.85% |
| 금융기관사칭 | 11 | 8.53% |
| 기타피싱 | 12 | 9.30% |
| 이벤트당첨사칭 | 17 | 13.18% |
| 일상대화 | 33 | 25.58% |
| 정부공공기관사칭 | 3 | 2.33% |
| 정상알림톡 | 37 | 28.68% |
| 정상포인트소멸알림 | 1 | 0.78% |
| 택배사칭 | 1 | 0.78% |

### Test

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 13 | 10.32% |
| 금융기관사칭 | 6 | 4.76% |
| 기타피싱 | 8 | 6.35% |
| 대출_사기 | 2 | 1.59% |
| 이벤트당첨사칭 | 14 | 11.11% |
| 일상대화 | 32 | 25.40% |
| 정부공공기관사칭 | 4 | 3.17% |
| 정상알림톡 | 16 | 12.70% |
| 정상포인트소멸알림 | 3 | 2.38% |
| 지인사칭 | 21 | 16.67% |
| 택배사칭 | 7 | 5.56% |
