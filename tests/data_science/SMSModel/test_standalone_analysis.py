"""run_standalone_analysis.py의 selection split 폴백 로직 테스트"""
import numpy as np

from data_science.SMSModel.run_standalone_analysis import (
    SELECTION_SPLIT,
    resolve_selection_split,
)


def test_resolve_selection_split_prefers_train_oof_when_present() -> None:
    scored = {
        "train_oof": (np.array([0.1, 0.9]), np.array(["normal", "phishing"])),
        "validation": (np.array([0.2, 0.8]), np.array(["normal", "phishing"])),
    }

    assert resolve_selection_split(scored) == SELECTION_SPLIT == "train_oof"


def test_resolve_selection_split_falls_back_to_validation_for_legacy_artifacts() -> None:
    """oof_probabilities_ 없는 구버전 artifact는 validation으로 폴백해야 한다"""
    scored = {
        "validation": (np.array([0.2, 0.8]), np.array(["normal", "phishing"])),
        "test": (np.array([0.3, 0.7]), np.array(["normal", "phishing"])),
    }

    assert resolve_selection_split(scored) == "validation"
