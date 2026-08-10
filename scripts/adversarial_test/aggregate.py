import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
ORIGINAL_LABEL = "ORIGINAL"
EXCERPT_LENGTH = 60


def read_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["rule_detected"] = row["rule_detected"] == "True"
        row["pipeline_detected"] = row["pipeline_detected"] == "True"
        row["rule_score"] = int(row["rule_score"])
        row["pipeline_score"] = int(row["pipeline_score"])
    return rows


def _rate(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) * 100 if rows else 0.0


def _detection_row(label: str, rows: list[dict]) -> str:
    return (
        f"| {label} | {len(rows)} | {_rate(rows, 'rule_detected'):.1f}% "
        f"| {_rate(rows, 'pipeline_detected'):.1f}% |"
    )


def build_summary(rows: list[dict]) -> str:
    originals = [r for r in rows if r["mutation_type"] == ORIGINAL_LABEL]
    mutated = [r for r in rows if r["mutation_type"] != ORIGINAL_LABEL]

    lines = [
        "# SafeFam 사기문자 변형 공격 테스트 결과",
        "",
        "탐지 성공 기준: 규칙엔진 rule_score >= 40 / 파이프라인 risk_grade != LOW",
        "",
        "## 1. 원본 vs 변형 전체 탐지율",
        "",
        "| 구분 | 건수 | 규칙엔진 단독 탐지율 | 전체 파이프라인 탐지율 |",
        "| --- | ---: | ---: | ---: |",
        _detection_row("원본 (ORIGINAL)", originals),
        _detection_row("변형 전체", mutated),
        "",
        "## 2. 변형 유형별 탐지율 (규칙엔진 탐지율 낮은 순)",
        "",
        "| 변형 유형 | 건수 | 규칙엔진 탐지율 | 파이프라인 탐지율 | 규칙 평균점수 | 파이프라인 평균점수 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    by_mutation: dict[str, list[dict]] = {}
    for row in mutated:
        by_mutation.setdefault(row["mutation_type"], []).append(row)

    for name, group in sorted(by_mutation.items(), key=lambda item: _rate(item[1], "rule_detected")):
        rule_avg = sum(r["rule_score"] for r in group) / len(group)
        pipe_avg = sum(r["pipeline_score"] for r in group) / len(group)
        lines.append(
            f"| {name} | {len(group)} | {_rate(group, 'rule_detected'):.1f}% "
            f"| {_rate(group, 'pipeline_detected'):.1f}% | {rule_avg:.1f} | {pipe_avg:.1f} |"
        )

    # 데모 핵심: 규칙엔진은 뚫렸지만 문맥분석 포함 파이프라인은 여전히 잡아낸 사례
    saved = [r for r in mutated if not r["rule_detected"] and r["pipeline_detected"]]
    saved_rate = len(saved) / len(mutated) * 100 if mutated else 0.0
    lines += [
        "",
        "## 3. 규칙엔진 실패 -> 파이프라인 방어 성공 사례",
        "",
        f"변형 {len(mutated)}건 중 {len(saved)}건 ({saved_rate:.1f}%)이 "
        "규칙엔진을 우회했으나 파이프라인이 탐지했습니다.",
        "",
        "| 샘플 ID | 변형 유형 | 규칙점수 | 파이프라인 점수/등급 | 변형문 발췌 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in saved:
        excerpt = row["text"][:EXCERPT_LENGTH].replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {row['sample_id']} | {row['mutation_type']} | {row['rule_score']} "
            f"| {row['pipeline_score']} ({row['pipeline_grade']}) | {excerpt}... |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="변형 공격 평가 결과 집계")
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR / "results.csv")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "summary.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = read_results(args.results)
    summary = build_summary(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary, encoding="utf-8")
    logger.info("[Aggregate] %d건 집계 -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
