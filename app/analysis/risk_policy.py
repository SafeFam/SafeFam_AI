"""Shared risk-grade policies for text analyzers."""

RISK_HIGH_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 40


def determine_text_risk_grade(risk_score: int) -> str:
    """Map a 0-100 text risk score to the analyzer's public grade."""
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "DANGEROUS"
    if risk_score >= RISK_MEDIUM_THRESHOLD:
        return "SUSPICIOUS"
    return "SAFE"
