import logging
from typing import Tuple, Dict, Any, Optional
from app.dto.schemas import RiskGrade, ContributionBreakdown

logger = logging.getLogger(__name__)

class ScoringEngine:
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
        naive_bayes_score: Optional[int],
        llm_score: int,
        llm_available: bool
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
                return ScoringEngine.BOTH_ENGINES_UNAVAILABLE_FALLBACK_SCORE
            return llm_score
        if not llm_available:
            return naive_bayes_score
        return round(
            ScoringEngine.NAIVE_BAYES_WEIGHT * naive_bayes_score
            + ScoringEngine.LLM_WEIGHT * llm_score
        )

    @staticmethod
    def calculate_score(
        llm_score: int, # LLM(Gemini)이 반환한 위험도 점수 (0~100). 나이브 베이즈 단독 판정 시엔 그 점수와 동일
        is_url_malicious: bool, # 하이브리드 URL 엔진의 최종 악성 판정 여부
        url_risk_score: float,  # 하이브리드 URL 엔진이 계산한 위험도 점수 (0.0~1.0)
        rule_score: int,   # 로컬 규칙 기반 엔진(금융기관 DB, 금융 키워드, 계좌/카드번호 등)의 위험도 점수 (0~100)
        has_url: bool = True,  # 문자 본문에 URL이 있었는지 여부 -> 트랙 가중치 재배분 기준
        naive_bayes_score: Optional[int] = None,  # 1차 나이브 베이즈 위험도 점수 (미수행 시 None)
        llm_available: bool = True,  # Gemini 2차 검증이 정상적으로 수행되었는지 여부
        is_confirmed_malicious: bool = False  # GSB 블랙리스트 등재 / VT 다수 엔진 합의 / 로컬 도메인 룰 중 하나라도 확정된 경우
    ) -> Tuple[int, RiskGrade, ContributionBreakdown]:

        # 텍스트 문맥 점수와 하이브리드 URL 엔진의 결과값을 결합하여 최종 위험도를 산출
        text_track_score = ScoringEngine._combine_text_track_score(naive_bayes_score, llm_score, llm_available)

        if has_url:
            text_weight = ScoringEngine.TEXT_TRACK_WEIGHT_WITH_URL
            rules_weight = ScoringEngine.RULES_TRACK_WEIGHT_WITH_URL

            # URL 트랙 기여 점수 (만점 30점)
            if is_url_malicious or url_risk_score > 0:
                url_contrib = round(url_risk_score * ScoringEngine.URL_TRACK_WEIGHT * 100)
                url_contrib = min(max(url_contrib, 0), 30)
            else:
                url_contrib = 0
        else:
            # 채점할 URL 자체가 없으므로 URL 트랙은 성립하지 않음 -> LLM/규칙 트랙으로 재배분
            text_weight = ScoringEngine.TEXT_TRACK_WEIGHT_NO_URL
            rules_weight = ScoringEngine.RULES_TRACK_WEIGHT_NO_URL
            url_contrib = 0

        # 1. 텍스트 트랙 기여 점수 — 나이브 베이즈 + Gemini 하이브리드 결합, URL 유무에 따라 50% 또는 65%
        llm_contrib = round(text_track_score * text_weight)

        # 2. 로컬 규칙 기반 기여 점수 — 금융기관 DB/키워드/계좌·카드번호 룰 엔진 점수(0~100)를
        #    URL 유무에 따라 20% 또는 35% 배점으로 환산
        rules_contrib = round(rule_score * rules_weight)
        rules_contrib = min(max(rules_contrib, 0), round(100 * rules_weight))

        # 최종 위험도 점수 합산
        final_score = min(llm_contrib + url_contrib + rules_contrib, 100)

        # 최종 점수 기반 임계치 등급 분기
        if final_score >= 70:
            risk_grade = RiskGrade.HIGH
        elif final_score >= 40:
            risk_grade = RiskGrade.MEDIUM
        else:
            risk_grade = RiskGrade.LOW

        # 확정 악성 URL 하드 오버라이드: 텍스트 문맥이 아무리 평범해도 아래 중 하나라도 확정되면
        # 가중합으로 희석되지 않고 무조건 HIGH 처리 (GSB만큼 신뢰도가 낮은 VT 단독/소수 탐지는 제외).
        # - GSB 블랙리스트 실제 등재 확인
        # - VT 다수 엔진(임계치 이상) 동시 합의 탐지
        # - 로컬 도메인 룰(.ru, testsafebrowsing 등) 매치
        # 등급-점수 표기 일관성을 위해 점수도 HIGH 임계치(70점) 이상으로 끌어올림.
        if is_confirmed_malicious:
            if risk_grade != RiskGrade.HIGH:
                logger.warning(
                    f"[Scoring Engine] 확정 악성 URL 감지 -> HIGH 등급 강제 오버라이드 (원래 점수: {final_score})"
                )
            final_score = max(final_score, 70)
            risk_grade = RiskGrade.HIGH

        logger.info(
            f"[Scoring Engine] 통합 연산 완료 -> 최종 점수: {final_score} | 등급: {risk_grade} "
                f"(LLM: {llm_contrib}, Hybrid-URL: {url_contrib}, Rules: {rules_contrib})"
        )

        breakdown = ContributionBreakdown(
            llm=llm_contrib,
            hybrid_url=url_contrib,
            rules=rules_contrib
        )

        return final_score, risk_grade, breakdown