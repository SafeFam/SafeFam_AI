import pytest

from app.analysis.schemas import RiskGrade
from app.analysis.scoring import RiskScoringEngine

ScoringEngine = RiskScoringEngine


def test_combine_text_track_score_uses_llm_score_when_no_naive_bayes_ran():
    """naive_bayes_score가 None이면 (구버전 호출부 호환) llm_score를 그대로 사용해야 한다."""
    score = ScoringEngine._combine_text_track_score(None, 77, True)
    assert score == 77


def test_combine_text_track_score_trusts_naive_bayes_when_gemini_unavailable():
    """
    Gemini가 실패했을 때는 이미 의심 판정해 에스컬레이션한 나이브 베이즈 점수를
    그대로 신뢰해야 한다 (fail-safe — API 장애로 점수가 0으로 깎이면 안 됨).
    """
    score = ScoringEngine._combine_text_track_score(91, 0, False)
    assert score == 91


def test_combine_text_track_score_weights_gemini_higher_when_both_available():
    """둘 다 정상 수행된 경우 0.3(나이브베이즈) + 0.7(Gemini) 가중 평균으로 결합되어야 한다."""
    score = ScoringEngine._combine_text_track_score(91, 5, True)
    assert score == round(0.3 * 91 + 0.7 * 5)


def test_combine_text_track_score_does_not_fail_open_when_both_engines_unavailable():
    """
    나이브 베이즈 모델 로드 실패(naive_bayes_score=None) + Gemini 호출도 실패(llm_available=False)가
    동시에 발생한 경우, llm_score(보통 0)를 그대로 반환해 "안전"으로 둔갑시키면 안 되고
    fail-safe 대체 점수(SAFE 문턱 이상)를 반환해야 한다.
    """
    score = ScoringEngine._combine_text_track_score(None, 0, False)

    assert score == ScoringEngine.BOTH_ENGINES_UNAVAILABLE_FALLBACK_SCORE
    assert score >= 40  # SAFE로 오판되지 않도록 최소 MEDIUM 문턱 이상이어야 함


def test_calculate_score_both_engines_down_lands_at_least_medium_not_low():
    """
    두 텍스트 분류기가 모두 다운된 상태로 파이프라인 전체를 돌려도, URL/규칙 신호가 없다면
    최종 등급이 조용히 LOW로 나오면 안 되고 최소 MEDIUM 이상으로 판정되어야 한다.
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=0,
        is_url_malicious=False,
        url_risk_score=0.0,
        rule_score=0,
        naive_bayes_score=None,
        llm_available=False,
    )

    assert risk_grade != RiskGrade.LOW
    assert final_score >= 40


def test_calculate_score_naive_bayes_false_positive_is_dampened_by_gemini():
    """
    실제 확인된 나이브 베이즈 오탐 사례("엄마 오늘 저녁 메뉴 뭐야?" -> risk_score 97)를
    Gemini가 SAFE(0점)로 정확히 바로잡았을 때, 최종 점수가 나이브 베이즈 점수 하나로만
    계산했을 때보다 훨씬 낮게 나와야 한다 (하이브리드 결합이 오탐을 완화하는지 검증).
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=0,
        is_url_malicious=False,
        url_risk_score=0.0,
        rule_score=0,
        naive_bayes_score=97,
        llm_available=True,
    )

    assert final_score < 50
    assert breakdown.llm < 50


def test_calculate_score_backward_compatible_without_naive_bayes_args():
    """기존 호출부(나이브 베이즈 인자 없이 llm_score만 넘기는 방식)와 동일하게 동작해야 한다."""
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=90, is_url_malicious=True, url_risk_score=0.95, rule_score=0
    )

    assert breakdown.llm == 45


def test_calculate_score_gsb_confirmed_forces_high_even_with_benign_text_and_no_url_score():
    """
    GSB가 실제로 블랙리스트 등재를 확인한 경우, 텍스트 문맥이 완전히 평범하고
    다른 트랙 점수가 낮아도 최종 등급은 무조건 HIGH로 강제되어야 하고,
    점수도 HIGH 임계치(70점) 이상으로 끌어올려져야 한다.
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=5,  # 텍스트는 평범함
        is_url_malicious=True,
        url_risk_score=0.95,
        rule_score=0,
        is_confirmed_malicious=True,
    )

    assert risk_grade == RiskGrade.HIGH
    assert final_score >= 70


def test_calculate_score_confirmed_override_keeps_breakdown_reconciled_with_final_score():
    """
    확정 악성 오버라이드로 final_score가 70점 이상으로 강제 상향되면, 그 상승분이
    contribution_breakdown에도 반영되어 llm+hybrid_url+rules 합계가 final_score와
    일치해야 한다 (오버라이드 전 가중합 그대로 남아 점수와 어긋나면 안 됨).
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=5,  # 텍스트는 평범함 -> 오버라이드 전 가중합은 70점에 한참 못 미침
        is_url_malicious=True,
        url_risk_score=0.95,
        rule_score=0,
        is_confirmed_malicious=True,
    )

    assert risk_grade == RiskGrade.HIGH
    assert final_score == 70
    assert breakdown.llm + breakdown.hybrid_url + breakdown.rules == final_score
    # 오버라이드 사유가 URL 트랙의 확정 판정이므로 그쪽에 우선 배정되어야 한다.
    assert breakdown.hybrid_url == 30


def test_calculate_score_gsb_not_confirmed_does_not_trigger_override():
    """
    VT 단독 탐지 등 GSB 확정이 아닌 경우엔 오버라이드가 발동하지 않고
    기존 가중합 방식대로 등급이 매겨져야 한다.
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=5,
        is_url_malicious=True,
        url_risk_score=0.5,
        rule_score=0,
        is_confirmed_malicious=False,
    )

    assert risk_grade != RiskGrade.HIGH


def test_calculate_score_gsb_confirmed_does_not_lower_an_already_higher_score():
    """오버라이드는 점수를 70점 '이상'으로 보장할 뿐, 이미 그보다 높은 점수를 깎지 않아야 한다."""
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=95,
        is_url_malicious=True,
        url_risk_score=0.95,
        rule_score=100,
        is_confirmed_malicious=True,
    )

    assert final_score == min(round(95 * 0.5) + round(0.95 * 30) + 20, 100)
    assert risk_grade == RiskGrade.HIGH


def test_calculate_score_with_url_uses_50_30_20_weights():
    """URL이 있는 기존 케이스는 LLM 50% / URL 30% / 규칙 20% 그대로 유지되어야 한다."""
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=80,
        is_url_malicious=True,
        url_risk_score=0.5,
        rule_score=100,
        has_url=True,
    )

    assert breakdown.llm == round(80 * 0.5)  # 40
    assert breakdown.hybrid_url == round(0.5 * 30)  # 15
    assert breakdown.rules == 20


def test_calculate_score_without_url_reweights_to_65_35_and_url_track_is_zero():
    """
    URL이 없는 순수 텍스트 케이스는 URL 트랙(30%)이 LLM/규칙 트랙으로 재배분되어
    LLM 65% / 규칙 35%가 적용되고, URL 기여 점수는 항상 0이어야 한다.
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=80,
        is_url_malicious=False,
        url_risk_score=0.0,
        rule_score=100,
        has_url=False,
    )

    assert breakdown.llm == round(80 * 0.65)  # 52
    assert breakdown.hybrid_url == 0
    assert breakdown.rules == 35
    assert final_score == round(80 * 0.65) + 35


def test_calculate_score_without_url_raises_ceiling_above_old_50_point_cap():
    """
    URL 없는 순수 텍스트 스미싱이 예전엔 LLM 트랙 상한(50점)에 막혀 있었지만,
    재가중치 이후엔 65점까지 갈 수 있어야 한다 (여전히 70점 HIGH 문턱에는 못 미치지만
    규칙 트랙이 함께 반영되면 HIGH까지 갈 수 있는 여지가 생긴 것이 핵심).
    """
    final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
        llm_score=100,
        is_url_malicious=False,
        url_risk_score=0.0,
        rule_score=0,
        has_url=False,
    )

    assert breakdown.llm == 65
    assert final_score == 65
    assert final_score > 50


def test_redistributes_url_weight_when_url_unavailable():
    final_score, _, breakdown = RiskScoringEngine.calculate_score(
        llm_score=70,
        is_url_malicious=False,
        url_risk_score=0.0,
        rule_score=70,
        has_url=True,
        url_available=False,
    )

    assert breakdown.hybrid_url == 0
    assert breakdown.llm == 50
    assert breakdown.rules == 20
    assert final_score == 70


def test_raises_when_all_tracks_are_unavailable():
    with pytest.raises(
        ValueError,
        match="No analysis tracks are available",
    ):
        RiskScoringEngine.calculate_score(
            llm_score=0,
            is_url_malicious=False,
            url_risk_score=0.0,
            rule_score=0,
            has_url=True,
            text_available=False,
            url_available=False,
            rules_available=False,
        )
