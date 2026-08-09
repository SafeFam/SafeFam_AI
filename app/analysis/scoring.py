import logging

from app.analysis.schemas import ContributionBreakdown, RiskGrade

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    3중 가중치 스코어링 시스템
    - LLM 문맥 분석 — 나이브 베이즈(1차) + Gemini(2차) 하이브리드 결합 점수
    - 하이브리드 URL 보안 엔진 [GSB + VT 백업]
    - 로컬 가드 규칙 기반 패널티

    URL 포함 여부에 따라 트랙 가중치가 달라짐:
    - URL 있음: LLM 50% + 하이브리드 URL 30% + 규칙 20%
    - URL 없음: URL 트랙이 원천적으로 성립하지 않으므로(채점할 URL 자체가 없음)
      해당 30%를 LLM과 규칙 트랙에 재분배 -> LLM 65% + 규칙 35%
      (URL 없는 순수 텍스트형 스미싱이 URL 트랙 부재만으로 최종 등급이 낮게 깎이는 것을 방지)
    """

    # 텍스트 트랙 내부 하이브리드 결합 가중치 (나이브 베이즈 + Gemini 둘 다 수행된 경우에만 적용)
    NAIVE_BAYES_WEIGHT = 0.3
    LLM_WEIGHT = 0.7

    # 나이브 베이즈와 Gemini가 동시에 실패해 텍스트 트랙에 아무 신호도 없을 때의 대체 점수.
    # 0을 반환하면 "판정 불가"가 "안전 확인됨"으로 둔갑해 fail-open이 되므로, 최종 등급이
    # 최소 MEDIUM 이상이 되도록 강제한다. 텍스트 트랙 가중치가 가장 낮은 경우(URL 있음, 50%)에도
    # 다른 트랙 기여가 전혀 없이 이 값 하나만으로 final_score가 40점(HIGH/MEDIUM 문턱)에 닿아야 하므로
    # 80 이상이어야 함 (80 * 0.5 = 40).
    BOTH_ENGINES_UNAVAILABLE_FALLBACK_SCORE = 80

    # 파이프라인 전체가 예외로 죽어 어떤 트랙도 점수를 내지 못했을 때의 대체 점수.
    # 0점을 반환하면 "분석 실패"가 "안전 확인됨(LOW)"으로 둔갑하는 fail-open이 되므로,
    # 등급이 최소 MEDIUM(40점 이상, HIGH 미만)으로 나오도록 강제한다.
    PIPELINE_FAILURE_FALLBACK_SCORE = 50

    # 3중 트랙 간 가중치 (URL 유무에 따라 재배분)
    TEXT_TRACK_WEIGHT_WITH_URL = 0.50
    RULES_TRACK_WEIGHT_WITH_URL = 0.20
    URL_TRACK_WEIGHT = 0.30

    TEXT_TRACK_WEIGHT_NO_URL = 0.65
    RULES_TRACK_WEIGHT_NO_URL = 0.35

    @staticmethod
    def _combine_text_track_score(
        naive_bayes_score: int | None, llm_score: int, llm_available: bool
    ) -> int:
        """
        나이브 베이즈(1차) + Gemini(2차) 텍스트 위험도를 하나의 점수로 결합.
        - 나이브 베이즈가 SAFE로 판정해 Gemini를 스킵한 경우: naive_bayes_score가 곧 llm_score와 동일하므로 그대로 사용
        - 나이브 베이즈 모델 로드에 실패했지만 Gemini는 정상 수행된 경우: Gemini 점수를 그대로 신뢰
        - 나이브 베이즈와 Gemini가 동시에 실패한 경우: 신뢰할 수 있는 신호가 전혀 없으므로 llm_score(보통 0)를
          그대로 반환하지 않고 대체 점수로 fail-safe 처리 (두 분류기가 동시에 다운됐다는 이유만으로
          "안전"으로 오판되는 것을 방지)
        - Gemini 호출이 실패한 경우(나이브 베이즈는 정상): 나이브 베이즈가 이미 SAFE 미만(의심)으로 판단해
          에스컬레이션한 상황이므로, Gemini 장애를 이유로 점수를 0으로 깎지 않고 나이브 베이즈 점수를 그대로 신뢰
        - 둘 다 정상 수행된 경우: Gemini(문맥 분석)에 더 큰 가중치를 두고 나이브 베이즈 신호를 보조적으로 반영
        """
        if naive_bayes_score is None:
            if not llm_available:
                return RiskScoringEngine.BOTH_ENGINES_UNAVAILABLE_FALLBACK_SCORE
            return llm_score
        if not llm_available:
            return naive_bayes_score
        return round(
            RiskScoringEngine.NAIVE_BAYES_WEIGHT * naive_bayes_score
            + RiskScoringEngine.LLM_WEIGHT * llm_score
        )

    @staticmethod
    def _normalize_weights(
        text_weight: float,
        url_weight: float,
        rules_weight: float,
        *,
        text_available: bool,
        url_available: bool,
        rules_available: bool,
    ) -> tuple[float, float, float]:
        """사용 가능한 분석 트랙들의 가중치 합이 1.0(100%)이 되도록 정규화"""
        available_text_weight = text_weight if text_available else 0.0
        available_url_weight = url_weight if url_available else 0.0
        available_rules_weight = rules_weight if rules_available else 0.0

        total_weight = (
            available_text_weight + available_url_weight + available_rules_weight
        )

        if total_weight == 0:
            raise ValueError("No analysis tracks are available")

        return (
            available_text_weight / total_weight,
            available_url_weight / total_weight,
            available_rules_weight / total_weight,
        )

    @staticmethod
    def calculate_score(
        llm_score: int,
        is_url_malicious: bool,
        url_risk_score: float,
        rule_score: int,
        has_url: bool = True,
        naive_bayes_score: int | None = None,
        llm_available: bool = True,
        is_confirmed_malicious: bool = False,
        text_available: bool = True,
        url_available: bool = True,
        rules_available: bool = True,
    ) -> tuple[
        int,
        RiskGrade,
        ContributionBreakdown,
    ]:
        """모든 분석 트랙의 점수와 가중치를 종합하여 최종 점수, 위험 등급, 트랙별 기여도 계산"""
        text_track_score = RiskScoringEngine._combine_text_track_score(
            naive_bayes_score=naive_bayes_score,
            llm_score=llm_score,
            llm_available=llm_available,
        )
        effective_text_available = text_available or (
            naive_bayes_score is None and not llm_available
        )

        # URL 유무에 따른 기본 가중치 설정
        if has_url:
            text_weight = RiskScoringEngine.TEXT_TRACK_WEIGHT_WITH_URL
            url_weight = RiskScoringEngine.URL_TRACK_WEIGHT
            rules_weight = RiskScoringEngine.RULES_TRACK_WEIGHT_WITH_URL
        else:
            text_weight = RiskScoringEngine.TEXT_TRACK_WEIGHT_NO_URL
            url_weight = 0.0
            rules_weight = RiskScoringEngine.RULES_TRACK_WEIGHT_NO_URL
            url_available = False

        # 트랙별 가용성 상태를 반영한 가중치 정규화
        (
            text_weight,
            url_weight,
            rules_weight,
        ) = RiskScoringEngine._normalize_weights(
            text_weight=text_weight,
            url_weight=url_weight,
            rules_weight=rules_weight,
            text_available=effective_text_available,
            url_available=url_available,
            rules_available=rules_available,
        )

        # 텍스트 트랙 점수 기여도 계산
        if effective_text_available:
            llm_contrib = round(text_track_score * text_weight)
        else:
            llm_contrib = 0

        # URL 트랙 점수 기여도 계산
        if url_available:
            normalized_url_score = min(
                max(url_risk_score, 0.0),
                1.0,
            )
            url_contrib = round(normalized_url_score * url_weight * 100)
        else:
            url_contrib = 0

        # 룰 트랙 점수 기여도 계산
        if rules_available:
            normalized_rule_score = min(
                max(rule_score, 0),
                100,
            )
            rules_contrib = round(normalized_rule_score * rules_weight)
        else:
            rules_contrib = 0

        contribution_total = llm_contrib + url_contrib + rules_contrib

        # 반올림 오차로 인해 총합이 100점을 초과하는 경우 보정
        if contribution_total > 100:
            overflow = contribution_total - 100

            if llm_contrib >= max(
                url_contrib,
                rules_contrib,
            ):
                llm_contrib -= overflow
            elif url_contrib >= rules_contrib:
                url_contrib -= overflow
            else:
                rules_contrib -= overflow

        final_score = llm_contrib + url_contrib + rules_contrib

        # 점수 위험 등급 판정
        if final_score >= 70:
            risk_grade = RiskGrade.HIGH
        elif final_score >= 40:
            risk_grade = RiskGrade.MEDIUM
        else:
            risk_grade = RiskGrade.LOW

        # 확정 악성 신호 감지 시 최소 HIGH 등급(70점) 보장
        if is_confirmed_malicious:
            if risk_grade != RiskGrade.HIGH:
                logger.warning(
                    "[Scoring Engine] 확정 악성 신호 "
                    "감지: HIGH 등급으로 조정 "
                    "(기존 점수: %s)",
                    final_score,
                )

            target_score = max(final_score, 70)
            score_gap = target_score - final_score

            if score_gap > 0:
                url_capacity = max(
                    round(url_weight * 100) - url_contrib,
                    0,
                )
                url_add = min(
                    score_gap,
                    url_capacity,
                )
                url_contrib += url_add
                score_gap -= url_add

                rules_capacity = max(
                    round(rules_weight * 100) - rules_contrib,
                    0,
                )
                rules_add = min(
                    score_gap,
                    rules_capacity,
                )
                rules_contrib += rules_add
                score_gap -= rules_add

                text_capacity = max(
                    round(text_weight * 100) - llm_contrib,
                    0,
                )
                text_add = min(
                    score_gap,
                    text_capacity,
                )
                llm_contrib += text_add

            final_score = target_score
            risk_grade = RiskGrade.HIGH

        logger.info(
            "[Scoring Engine] 계산 완료 - "
            "점수: %s, 등급: %s "
            "(TEXT: %s, URL: %s, RULES: %s)",
            final_score,
            risk_grade,
            llm_contrib,
            url_contrib,
            rules_contrib,
        )

        breakdown = ContributionBreakdown(
            llm=llm_contrib,
            hybrid_url=url_contrib,
            rules=rules_contrib,
        )

        return final_score, risk_grade, breakdown
