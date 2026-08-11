import argparse
import csv
import hashlib
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path("data_science/Data/SMSData/phishing_total_dataset_2705.csv")
OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "corpus.json"

# NB 학습 파이프라인(train_sms.py)이 실제로 학습에 쓴 문장의 fingerprint 집합. 벤치마크
# 코퍼스가 이 문장들과 겹치면 "이미 외운 문제로 시험 본" 격이라 정확도가 부풀려진다.
_SPLIT_MANIFEST_PATH = Path("data_science/SMSModel/splits/sms_split_v1.csv")


def _load_train_fingerprints() -> set[str]:
    from data_science.SMSModel.train_sms import DATA_PATH as TRAIN_DATA_PATH
    from data_science.SMSModel.train_sms import load_data

    df_pool, _df_holdout = load_data(TRAIN_DATA_PATH)

    train_fingerprints: set[str] = set()
    with _SPLIT_MANIFEST_PATH.open(encoding="utf-8", newline="") as f:
        manifest = {row["text_fingerprint"]: row["split"] for row in csv.DictReader(f)}

    for fingerprint in df_pool["text_fingerprint"]:
        if manifest.get(fingerprint) == "train":
            train_fingerprints.add(fingerprint)

    return train_fingerprints


def _fingerprint(text: str) -> str:
    from app.analysis.text.preprocessing import normalize_text

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# label(phishing/normal) 안에서 다시 type별로 층화추출 — 특정 유형(예: 정상알림톡)에
# 표본이 쏠려서 오탐률이 왜곡되는 것을 방지한다
def _sample_label(
    rows_by_type: dict[str, list[dict]],
    label: str,
    total: int,
    rng: random.Random,
) -> list[dict]:
    types = sorted(rows_by_type)
    if not types:
        return []

    per_type = max(total // len(types), 1)
    sampled: list[dict] = []

    for sms_type in types:
        rows = rows_by_type[sms_type]
        take = min(per_type, len(rows))
        for row in rng.sample(rows, take):
            sampled.append({**row, "label": label, "type": sms_type})

    # 균등분배 후 부족분은 전체 잔여 풀에서 채워 목표 건수에 최대한 맞춘다
    if len(sampled) < total:
        used_texts = {item["text"] for item in sampled}
        remaining = [
            {**row, "label": label, "type": sms_type}
            for sms_type in types
            for row in rows_by_type[sms_type]
            if row["text"] not in used_texts
        ]
        rng.shuffle(remaining)
        sampled.extend(remaining[: total - len(sampled)])

    return sampled[:total]


# 성능 벤치마크용 코퍼스: phishing/normal 라벨을 균형있게 뽑아 탐지율(Recall)과
# 오탐률(FPR)을 동시에 측정할 수 있게 한다 (adversarial_test/corpus.py는 phishing만 다룸)
def select_corpus(
    csv_path: Path,
    phishing_count: int,
    normal_count: int,
    seed: int = 42,
    clean_only: bool = False,
) -> list[dict]:
    train_fingerprints = _load_train_fingerprints() if clean_only else set()
    excluded = 0

    by_label_type: dict[str, dict[str, list[dict]]] = {"phishing": {}, "normal": {}}

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            text = (row.get("text") or "").strip()
            sms_type = (row.get("type") or "").strip()
            if label not in by_label_type or not text or not sms_type:
                continue
            # 이미 신규 홀드아웃(source=synthetic_*)인 행은 애초에 train_fingerprints에
            # 없으므로 자동으로 통과된다 — fingerprint 하나로 두 케이스를 함께 처리
            if clean_only and _fingerprint(text) in train_fingerprints:
                excluded += 1
                continue
            by_label_type[label].setdefault(sms_type, []).append(
                {"text": text, "has_url": (row.get("has_url") or "").strip().lower() == "true"}
            )

    if clean_only:
        logger.info("[BenchCorpus] clean_only: 학습셋과 겹치는 %d건 제외", excluded)

    rng = random.Random(seed)
    phishing = _sample_label(by_label_type["phishing"], "phishing", phishing_count, rng)
    normal = _sample_label(by_label_type["normal"], "normal", normal_count, rng)

    corpus: list[dict] = []
    for idx, row in enumerate(phishing):
        corpus.append({"id": f"phishing_{idx}", **row})
    for idx, row in enumerate(normal):
        corpus.append({"id": f"normal_{idx}", **row})

    logger.info(
        "[BenchCorpus] phishing %d건 / normal %d건 (요청: %d/%d)",
        len(phishing),
        len(normal),
        phishing_count,
        normal_count,
    )
    return corpus


def write_corpus(corpus: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[BenchCorpus] %d건 저장 -> %s", len(corpus), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="성능 벤치마크용 phishing/normal 균형 코퍼스 생성")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--phishing-count", type=int, default=200)
    parser.add_argument("--normal-count", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="NB 학습셋(train split)과 겹치는 문장을 제외하고 코퍼스 생성",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = select_corpus(args.csv, args.phishing_count, args.normal_count, clean_only=args.clean_only)
    write_corpus(corpus, args.out)


if __name__ == "__main__":
    main()
