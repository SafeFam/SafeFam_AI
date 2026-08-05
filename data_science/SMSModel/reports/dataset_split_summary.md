# SMS Dataset Split Summary

## Dataset

- Schema version: `1`
- Dataset fingerprint: `1db45d5f2c3d3d17726888b05cd625e0d0a51deef3dc8ab94016a9ee97af18f5`
- Total rows: 817
- Template groups: 539

## Configuration

- Template similarity threshold: `0.88`
- Template n-gram range: `[2, 5]`
- Random state: `42`

## Split Overview

| Split | Rows | Ratio | Groups | Normal | Phishing |
|---|---:|---:|---:|---:|---:|
| train | 571 | 69.89% | 377 | 261 (45.71%) | 310 (54.29%) |
| validation | 123 | 15.06% | 81 | 56 (45.53%) | 67 (54.47%) |
| test | 123 | 15.06% | 81 | 56 (45.53%) | 67 (54.47%) |

## Leakage Validation

- Passed: `True`
- Template group overlap count: `0`
- Fingerprint overlap count: `0`

## Message Type Distribution

### Train

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 56 | 9.81% |
| 금융기관사칭 | 49 | 8.58% |
| 기타피싱 | 50 | 8.76% |
| 이벤트당첨사칭 | 72 | 12.61% |
| 일상대화 | 103 | 18.04% |
| 정부공공기관사칭 | 18 | 3.15% |
| 정상알림톡 | 158 | 27.67% |
| 지인사칭 | 43 | 7.53% |
| 택배사칭 | 22 | 3.85% |

### Validation

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 4 | 3.25% |
| 금융기관사칭 | 24 | 19.51% |
| 기타피싱 | 22 | 17.89% |
| 이벤트당첨사칭 | 5 | 4.06% |
| 일상대화 | 26 | 21.14% |
| 정부공공기관사칭 | 1 | 0.81% |
| 정상알림톡 | 30 | 24.39% |
| 지인사칭 | 1 | 0.81% |
| 택배사칭 | 10 | 8.13% |

### Test

| Type | Count | Ratio |
|---|---:|---:|
| 경조사사칭 | 8 | 6.50% |
| 금융기관사칭 | 17 | 13.82% |
| 기타피싱 | 9 | 7.32% |
| 이벤트당첨사칭 | 16 | 13.01% |
| 일상대화 | 26 | 21.14% |
| 정부공공기관사칭 | 5 | 4.06% |
| 정상알림톡 | 30 | 24.39% |
| 지인사칭 | 2 | 1.63% |
| 택배사칭 | 10 | 8.13% |
