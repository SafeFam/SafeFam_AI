# SMS Model Workspace

| Path | Purpose |
|---|---|
| `train_sms.py` | Naive Bayes training entry point |
| `artifacts/` | Versioned model and vectorizer files used by the API |
| `dataset_splitting/` | Leakage-safe train/validation/test splitting |
| `template_grouping/` | Duplicate and similar-message grouping |
| `splits/` | Reproducible split manifests |
| `reporting/` | Dataset report generation code |
| `reports/` | Generated summaries, metrics, and feature analysis data |
| `reports/figures/` | Generated plots and figures |
| `SMSDataModel.ipynb` | Exploratory analysis notebook |

Run training from the repository root:

```bash
python -m data_science.SMSModel.train_sms
```

The committed split manifest fixes the final test set. Do not overwrite it during
routine training. Create a new manifest version when the dataset, preprocessing,
template grouping, or split policy intentionally changes.
