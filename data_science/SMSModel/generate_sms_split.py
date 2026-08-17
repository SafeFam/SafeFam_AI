"""검수된 SMS 데이터로 새 split manifest와 보고서를 생성"""

from __future__ import annotations

from data_science.SMSModel.train_sms import (
    DATA_PATH,
    SPLIT_MANIFEST_PATH,
    load_data,
    split_data,
)


def main() -> None:
    dataset, holdout = load_data(DATA_PATH)

    splits = split_data(
        dataset,
        create_manifest=True,
    )

    print(
        "[SMS split] generated "
        f"train={len(splits.train)} "
        f"validation={len(splits.validation)} "
        f"test={len(splits.test)} "
        f"holdout={len(holdout)}"
    )
    print(
        f"[SMS split] manifest="
        f"{SPLIT_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
