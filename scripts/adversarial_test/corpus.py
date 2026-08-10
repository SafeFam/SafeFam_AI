import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path

from app.analysis.rules.analyzer import analyze_text_with_rules

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path("data_science/Data/SMSData/phishing_total_dataset_2705.csv")
OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "corpus.json"

_RE_NON_SLUG = re.compile(r"[^0-9a-zA-Z가-힣]+")


# 파일명/ID로 쓸 수 있게 카테고리 한글명을 정규화 (한글은 그대로 두고 공백/기호만 제거)
def _slugify(value: str) -> str:
    return _RE_NON_SLUG.sub("_", value).strip("_") or "unknown"


# "금융기관사칭=25,정부공공기관사칭=10" 형태를 카테고리별 목표 건수 dict로 변환
def parse_category_overrides(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    overrides: dict[str, int] = {}
    for pair in raw.split(","):
        name, _, count = pair.partition("=")
        overrides[name.strip()] = int(count)
    return overrides


# 사기문자만 골라 카테고리별 샘플링. category_overrides에 없는 카테고리는 per_type을 기본값으로 사용 —
# 특정 시나리오(예: 금융기관사칭)를 집중 검증하면서도 다른 유형과의 비교 기준선은 남겨둔다
def select_corpus(
    csv_path: Path,
    per_type: int = 6,
    seed: int = 42,
    category_overrides: dict[str, int] | None = None,
    keyword_filter: list[str] | None = None,
    min_rule_score: int | None = None,
) -> list[dict]:
    by_type: dict[str, list[dict]] = {}

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label") != "phishing":
                continue
            sms_type = (row.get("type") or "").strip()
            text = (row.get("text") or "").strip()
            if not sms_type or not text:
                continue
            # 키워드 필터: "계좌 정지형" 등 특정 사기 시나리오만 골라 코퍼스를 좁힐 때 사용
            if keyword_filter and not any(kw in text for kw in keyword_filter):
                continue
            # min_rule_score: 변형 실험의 "원본은 규칙엔진이 잡는다"는 전제를 보장하려면 느슨한
            # 키워드 매칭보다 실제 규칙엔진 점수로 직접 거르는 편이 정확하다
            if min_rule_score is not None and analyze_text_with_rules(text)["rule_score"] < min_rule_score:
                continue
            by_type.setdefault(sms_type, []).append(
                {"text": text, "has_url": (row.get("has_url") or "").strip().lower() == "true"}
            )

    rng = random.Random(seed)
    corpus: list[dict] = []
    overrides = category_overrides or {}

    for sms_type in sorted(by_type):
        rows = by_type[sms_type]
        target = overrides.get(sms_type, per_type)
        sampled = rng.sample(rows, min(target, len(rows)))
        slug = _slugify(sms_type)
        for idx, row in enumerate(sampled):
            corpus.append(
                {
                    "id": f"{slug}_{idx}",
                    "text": row["text"],
                    "type": sms_type,
                    "has_url": row["has_url"],
                }
            )
        logger.info("[Corpus] %s: %d개 샘플링 (전체 %d개)", sms_type, len(sampled), len(rows))

    return corpus


def write_corpus(corpus: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[Corpus] %d건 저장 -> %s", len(corpus), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="사기문자 코퍼스 생성")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument(
        "--category-overrides",
        type=str,
        default=None,
        help="예: 금융기관사칭=25,정부공공기관사칭=10",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="쉼표 구분 키워드. 지정 시 텍스트에 하나라도 포함된 행만 사용 (예: 계좌,이체,정지,동결)",
    )
    parser.add_argument(
        "--min-rule-score",
        type=int,
        default=None,
        help="지정 시 규칙엔진 점수가 이 값 이상인 원문만 사용 (원본은 규칙엔진이 이미 탐지하는 케이스로 한정)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = select_corpus(
        args.csv,
        per_type=args.per_type,
        category_overrides=parse_category_overrides(args.category_overrides),
        keyword_filter=args.keywords.split(",") if args.keywords else None,
        min_rule_score=args.min_rule_score,
    )
    write_corpus(corpus, args.out)


if __name__ == "__main__":
    main()
