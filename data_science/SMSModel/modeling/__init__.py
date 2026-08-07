"""SMS 피싱 모델 공통 API"""
from .base import(
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType
)

__all__ = [
    "BasePhishingClassifier",
    "ScoreOutput",
    "ScoreType",
]

