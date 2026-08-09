"""Shared fixtures for SMS model classifier tests."""

import pandas as pd
import pytest


@pytest.fixture
def training_dataframe():
    """Return a balanced dataset large enough for calibrated NB tests."""
    rows = []

    for index in range(20):
        rows.append(
            {
                "text": f"오늘 회의 시간 안내 {index}",
                "text_norm": f"오늘 회의 시간 안내 {index}",
                "has_url": False,
                "label": "normal",
            }
        )
        rows.append(
            {
                "text": f"계좌 정지 확인 필요 http://bit.ly/fake{index}",
                "text_norm": "계좌 정지 확인 필요 [URL]",
                "has_url": True,
                "label": "phishing",
            }
        )

    return pd.DataFrame(rows)
