import argparse
import json
import logging
from pathlib import Path

from app.analysis.text.llm_analyzer import SYSTEM_PROMPT
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# 프로덕션 llm_analyzer.py의 SYSTEM_PROMPT를 그대로 재사용해 챗 앱 프록시 판정이
# 실제 API 호출과 최대한 같은 기준으로 이뤄지게 한다. 배치 처리를 위해 출력 스키마만
# 단일 텍스트용 RESPONSE_SCHEMA 대신 {id: risk_score} 형태로 바꿔 요청한다.
_INSTRUCTIONS = """{system_prompt}

--- 배치 처리 지시 ---
위 기준을 그대로 적용해서, 아래 문자 목록 각각에 대해 risk_score(0~100 정수)만 산정해줘.
다른 설명 없이 아래 JSON 형식 그대로만 출력해줘 (마크다운 코드펜스도 붙이지 마):

{{
  "<id>": <risk_score 정수>,
  ...
}}

문자 목록 ({count}건):
"""


def _format_batch(batch: list[dict]) -> str:
    lines = [_INSTRUCTIONS.format(system_prompt=SYSTEM_PROMPT, count=len(batch))]
    for item in batch:
        lines.append(f"\n[{item['id']}]\n{item['text']}")
    return "".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM만 트랙 정확도 측정용 챗 앱 프록시 프롬프트 생성")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    batch_count = 0
    for start in range(0, len(corpus), args.batch_size):
        batch = corpus[start : start + args.batch_size]
        batch_count += 1
        out_path = args.out_dir / f"llm_proxy_prompt_batch{batch_count}.txt"
        out_path.write_text(_format_batch(batch), encoding="utf-8")
        logger.info("[LlmProxyPrompt] batch%d (%d건) -> %s", batch_count, len(batch), out_path)

    logger.info("[LlmProxyPrompt] 총 %d개 배치 파일 생성 (전체 %d건)", batch_count, len(corpus))


if __name__ == "__main__":
    main()
