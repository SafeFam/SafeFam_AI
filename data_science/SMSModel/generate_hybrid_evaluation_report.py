"""저장된 평가 레코드에서 JSON·Markdown 비교 보고서를 생성합니다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_science.SMSModel.hybrid_evaluation.reporting import (
    build_comparison_report,
    render_markdown_report,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_RECORDS_PATH = (
    SMS_MODEL_DIRECTORY / "reports/hybrid_evaluation/evaluation_records.json"
)
DEFAULT_POLICY_PATH = (
    SMS_MODEL_DIRECTORY / "artifacts/stacking/hybrid_policy.json"
)
DEFAULT_JSON_PATH = (
    SMS_MODEL_DIRECTORY / "reports/hybrid_evaluation/comparison_report.json"
)
DEFAULT_MARKDOWN_PATH = (
    SMS_MODEL_DIRECTORY / "reports/hybrid_evaluation/comparison_report.md"
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def generate_reports(
    *,
    records_path: Path,
    policy_path: Path,
    json_path: Path,
    markdown_path: Path,
    input_price_per_million: float,
    output_price_per_million: float,
    currency: str,
    pricing_as_of: str,
) -> dict[str, Any]:
    source = _load_json(records_path)
    policy = _load_json(policy_path)
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("evaluation records must be a list")
    if source.get("record_count") != len(records):
        raise ValueError("evaluation record_count does not match records")

    report = build_comparison_report(
        records,
        source_metadata=source,
        policy=policy,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        currency=currency,
        pricing_as_of=pricing_as_of,
    )
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the offline Stacking/Claude/Hybrid report."
    )
    parser.add_argument("--records-path", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument(
        "--markdown-output", type=Path, default=DEFAULT_MARKDOWN_PATH
    )
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--pricing-as-of", required=True)
    arguments = parser.parse_args()

    report = generate_reports(
        records_path=arguments.records_path,
        policy_path=arguments.policy_path,
        json_path=arguments.json_output,
        markdown_path=arguments.markdown_output,
        input_price_per_million=arguments.input_price_per_million,
        output_price_per_million=arguments.output_price_per_million,
        currency=arguments.currency,
        pricing_as_of=arguments.pricing_as_of,
    )
    print(
        "[Hybrid report] completed "
        f"samples={report['sample_count']} json={arguments.json_output} "
        f"markdown={arguments.markdown_output}"
    )


if __name__ == "__main__":
    main()
