# Phishing Model Evaluation

| Model | Threshold | Precision | Recall | F1 | F2 | FN | FP | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_bayes_structural | 0.056923 | 0.5447 | 1.0000 | 0.7053 | 0.8568 | 0 | 56 | 4.003 | 5.414 |
| logistic_regression_morph_tfidf | 0.369065 | 0.7386 | 0.9701 | 0.8387 | 0.9129 | 2 | 23 | 4.918 | 18.899 |
| linear_svm_char_tfidf | -0.322302 | 0.7471 | 0.9701 | 0.8442 | 0.9155 | 2 | 22 | 1.596 | 2.529 |
