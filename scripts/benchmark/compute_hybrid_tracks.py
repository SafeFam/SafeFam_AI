import argparse
import csv
import json
import logging
import time
from pathlib import Path

from app.analysis.scoring import RiskScoringEngine
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH
from scripts.benchmark.metrics import compute_metrics, format_table

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DETECTION_THRESHOLD = 40


def _read_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


# local_tracks_raw.csv는 같은 id가 track별로(규칙만/NB만) 두 번 나온다 —
# id 하나로만 dict를 만들면 나중 track이 앞 track을 덮어써서 조용히 데이터가 사라진다.
def _read_local_tracks(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rules = {r["id"]: r for r in rows if r["track"] == "규칙만"}
    nb = {r["id"]: r for r in rows if r["track"] == "NB만"}
    return rules, nb


# _analyze_text_hybrid(service.py)와 동일한 스킵 로직 재현: NB가 SAFE로 확신하면
# Gemini 점수를 쓰지 않고 NB 점수를 그대로 텍스트 트랙 점수로 사용한다
def _hybrid_text_score(nb_score: int, llm_score: int) -> int:
    if nb_score < DETECTION_THRESHOLD:
        return nb_score
    return RiskScoringEngine._combine_text_track_score(
        naive_bayes_score=nb_score, llm_score=llm_score, llm_available=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="실측된 규칙/NB/LLM/URL 점수를 조합해 NB+LLM, SafeFam 전체 트랙 산출")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--local", type=Path, default=OUTPUT_DIR / "local_tracks_raw.csv")
    parser.add_argument("--llm", type=Path, default=OUTPUT_DIR / "llm_track_raw.csv")
    parser.add_argument("--url", type=Path, default=OUTPUT_DIR / "url_track_raw_full.csv")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "hybrid_tracks_raw.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = {c["id"]: c for c in json.loads(args.corpus.read_text(encoding="utf-8"))}
    rule_scores, nb_scores = _read_local_tracks(args.local)
    llm_rows = _read_csv(args.llm)
    url_rows = _read_csv(args.url) if args.url.exists() else {}

    missing = [sid for sid in llm_rows if sid not in rule_scores or sid not in nb_scores]
    if missing:
        logger.warning("[Hybrid] 규칙/NB 점수가 없어 제외된 샘플 %d건 (예: %s)", len(missing), missing[:3])

    hybrid_rows = []
    full_rows = []

    for sample_id, llm_row in llm_rows.items():
        if sample_id not in rule_scores or sample_id not in nb_scores:
            continue

        sample = corpus[sample_id]
        rule_score = int(rule_scores[sample_id]["score"])
        nb_score = int(nb_scores[sample_id]["score"])
        llm_score = int(llm_row["score"])
        label = llm_row["label"]

        text_score = _hybrid_text_score(nb_score, llm_score)
        hybrid_rows.append(
            {
                "id": sample_id,
                "label": label,
                "track": "NB+LLM",
                "score": text_score,
                "detected": text_score >= DETECTION_THRESHOLD,
                "elapsed_seconds": 0.0,
            }
        )

        has_url = sample["has_url"]
        url_row = url_rows.get(sample_id) if has_url else None
        is_url_malicious = url_row["is_malicious"] == "True" if url_row else False
        url_risk_score = float(url_row["url_risk_score"]) if url_row else 0.0
        url_available = has_url and url_row is not None and url_row["available"] == "True"

        final_score, risk_grade, _ = RiskScoringEngine.calculate_score(
            llm_score=text_score,
            is_url_malicious=is_url_malicious,
            url_risk_score=url_risk_score,
            rule_score=rule_score,
            has_url=has_url,
            naive_bayes_score=None,  # text_score에 이미 NB+LLM 결합 반영됨
            llm_available=True,
            is_confirmed_malicious=is_url_malicious,
            url_available=url_available,
        )
        full_rows.append(
            {
                "id": sample_id,
                "label": label,
                "track": "SafeFam 전체",
                "score": final_score,
                "detected": risk_grade != "LOW",
                "elapsed_seconds": 0.0,
            }
        )

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "track", "score", "detected", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(hybrid_rows + full_rows)

    llm_metrics = compute_metrics(
        "LLM만",
        [
            {"label": r["label"], "detected": r["detected"] == "True", "elapsed_seconds": float(r["elapsed_seconds"])}
            for r in llm_rows.values()
        ],
    )
    hybrid_metrics = compute_metrics(
        "NB+LLM", [{"label": r["label"], "detected": r["detected"], "elapsed_seconds": 0.0} for r in hybrid_rows]
    )
    full_metrics = compute_metrics(
        "SafeFam 전체", [{"label": r["label"], "detected": r["detected"], "elapsed_seconds": 0.0} for r in full_rows]
    )

    table = format_table([llm_metrics, hybrid_metrics, full_metrics])
    print(table)
    (OUTPUT_DIR / "hybrid_tracks_summary.md").write_text(table + "\n", encoding="utf-8")
    logger.info("[Hybrid] %d건 처리 -> %s", len(hybrid_rows), args.out)


if __name__ == "__main__":
    main()
