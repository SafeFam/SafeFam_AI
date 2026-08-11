from dataclasses import dataclass


@dataclass
class TrackMetrics:
    track: str
    total: int
    phishing_total: int
    normal_total: int
    tp: int
    fn: int
    fp: int
    tn: int
    recall: float
    fpr: float
    precision: float
    f1: float
    avg_time_seconds: float


# Recall(탐지율)은 phishing 대비 TP, FPR(오탐률)은 normal 대비 FP로 계산 —
# 두 라벨을 같은 잣대(detected: bool)로 평가해야 트랙 간 비교가 성립한다
def compute_metrics(track: str, rows: list[dict]) -> TrackMetrics:
    phishing = [r for r in rows if r["label"] == "phishing"]
    normal = [r for r in rows if r["label"] == "normal"]

    tp = sum(1 for r in phishing if r["detected"])
    fn = len(phishing) - tp
    fp = sum(1 for r in normal if r["detected"])
    tn = len(normal) - fp

    recall = tp / len(phishing) if phishing else 0.0
    fpr = fp / len(normal) if normal else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_time = sum(r["elapsed_seconds"] for r in rows) / len(rows) if rows else 0.0

    return TrackMetrics(
        track=track,
        total=len(rows),
        phishing_total=len(phishing),
        normal_total=len(normal),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=recall,
        fpr=fpr,
        precision=precision,
        f1=f1,
        avg_time_seconds=avg_time,
    )


def format_table(all_metrics: list[TrackMetrics]) -> str:
    lines = [
        "| 트랙 | 표본 (사기/정상) | 탐지율(Recall) | 오탐률(FPR) | F1 | 평균 분석시간 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m.track} | {m.phishing_total}/{m.normal_total} "
            f"| {m.recall * 100:.1f}% | {m.fpr * 100:.1f}% "
            f"| {m.f1:.3f} | {m.avg_time_seconds * 1000:.0f}ms |"
        )
    return "\n".join(lines)
