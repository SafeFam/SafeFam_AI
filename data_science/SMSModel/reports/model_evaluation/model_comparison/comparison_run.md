# Phishing Model Comparison

## Reproducibility

- Dataset fingerprint: `1db45d5f2c3d3d17726888b05cd625e0d0a51deef3dc8ab94016a9ee97af18f5`
- Split manifest: `sms_split_v1`
- Split manifest SHA-256: `43a7c2ab76f2521bce300a70ed1e61d176226185fa5e4702b796d8d6620ea847`
- Train rows: 571
- Validation rows: 123
- Test rows: 123
- Target phishing recall: 0.96
- Threshold selection: validation only
- Final classification metrics: test only
- Test-based hyperparameter tuning: disabled

## Model Evaluation

| Model | Threshold | Precision | Recall | F1 | F2 | FN | FP | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_bayes_structural | 0.056923 | 0.5447 | 1.0000 | 0.7053 | 0.8568 | 0 | 56 | 4.003 | 5.414 |
| logistic_regression_morph_tfidf | 0.369065 | 0.7386 | 0.9701 | 0.8387 | 0.9129 | 2 | 23 | 4.918 | 18.899 |
| linear_svm_char_tfidf | -0.322302 | 0.7471 | 0.9701 | 0.8442 | 0.9155 | 2 | 22 | 1.596 | 2.529 |

## Validation Threshold Selection

| Model | Threshold | Target Recall | Validation Recall | Target Met |
|---|---:|---:|---:|:---:|
| naive_bayes_structural | 0.056923 | 0.9600 | 1.0000 | yes |
| logistic_regression_morph_tfidf | 0.369065 | 0.9600 | 1.0000 | yes |
| linear_svm_char_tfidf | -0.322302 | 0.9600 | 1.0000 | yes |

## Test Confusion Matrices

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| naive_bayes_structural | 0 | 56 | 0 | 67 |
| logistic_regression_morph_tfidf | 33 | 23 | 2 | 65 |
| linear_svm_char_tfidf | 34 | 22 | 2 | 65 |

전체 모델 및 vectorizer 설정은 `comparison_run.json`의 `models[].metadata`에서 확인할 수 있습니다.
