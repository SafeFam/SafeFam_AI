import pandas as pd
import pytest

from data_science.SMSModel.template_grouping import (
    TemplateGroupingConfig,
    add_text_fingerprints,
    assign_template_groups,
    create_text_fingerprint,
    prepare_template_groups,
    remove_exact_duplicates,
    validate_duplicate_labels,
)


def make_dataframe(
    rows: list[tuple[str, str]],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "text_norm": text_norm,
                "label": label,
            }
            for text_norm, label in rows
        ]
    )


def test_create_text_fingerprint_is_deterministic():
    first = create_text_fingerprint("[URL]에서 건강검진 결과를 확인하세요")
    second = create_text_fingerprint("[URL]에서 건강검진 결과를 확인하세요")

    assert first == second
    assert len(first) == 64


def test_create_text_fingerprint_ignores_outer_whitespace():
    plain = create_text_fingerprint("본인 인증이 필요합니다")
    padded = create_text_fingerprint("  본인 인증이 필요합니다  ")

    assert plain == padded


def test_different_texts_have_different_fingerprints():
    first = create_text_fingerprint("배송 조회가 필요합니다")
    second = create_text_fingerprint("결제 확인이 필요합니다")

    assert first != second


def test_add_text_fingerprints_does_not_modify_original_dataframe():
    original = make_dataframe(
        [
            ("배송 조회가 필요합니다", "normal"),
        ]
    )

    result = add_text_fingerprints(original)

    assert "text_fingerprint" not in original.columns
    assert "text_fingerprint" in result.columns


def test_remove_exact_duplicates_keeps_one_row():
    df = make_dataframe(
        [
            ("[URL]에서 본인 인증하세요", "phishing"),
            ("[URL]에서 본인 인증하세요", "phishing"),
            ("회의는 오후 세 시입니다", "normal"),
        ]
    )
    df = add_text_fingerprints(df)

    result = remove_exact_duplicates(df)

    assert len(result) == 2
    assert result["text_norm"].tolist() == [
        "[URL]에서 본인 인증하세요",
        "회의는 오후 세 시입니다",
    ]


def test_validate_duplicate_labels_rejects_conflicting_labels():
    df = make_dataframe(
        [
            ("[URL]에서 본인 인증하세요", "phishing"),
            ("[URL]에서 본인 인증하세요", "normal"),
        ]
    )
    df = add_text_fingerprints(df)

    with pytest.raises(
        ValueError,
        match="conflicting labels",
    ):
        validate_duplicate_labels(df)


def test_similar_messages_receive_same_template_group():
    df = make_dataframe(
        [
            (
                "[국민건강보험] 건강검진 결과가 발급되었습니다 "
                "[URL]에서 즉시 확인하세요",
                "phishing",
            ),
            (
                "[국민건강보험] 건강검진 보고서가 발급되었습니다 "
                "[URL]에서 지금 확인하세요",
                "phishing",
            ),
            (
                "오늘 저녁 식사는 일곱 시에 시작합니다",
                "normal",
            ),
        ]
    )

    config = TemplateGroupingConfig(
        similarity_threshold=0.70,
        ngram_range=(2, 4),
        min_df=1,
        max_features=None,
    )

    result = prepare_template_groups(
        df,
        config=config,
    )

    first_group = result.iloc[0]["template_group_id"]
    second_group = result.iloc[1]["template_group_id"]
    unrelated_group = result.iloc[2]["template_group_id"]

    assert first_group == second_group
    assert first_group != unrelated_group


def test_unrelated_messages_receive_different_template_groups():
    df = make_dataframe(
        [
            (
                "[URL]에서 택배 배송 주소를 확인하세요",
                "phishing",
            ),
            (
                "회의가 오후 세 시로 변경되었습니다",
                "normal",
            ),
            (
                "저녁 식사 재료를 구매했습니다",
                "normal",
            ),
        ]
    )

    config = TemplateGroupingConfig(
        similarity_threshold=0.88,
    )

    result = prepare_template_groups(
        df,
        config=config,
    )

    assert result["template_group_id"].nunique() == 3


def test_group_id_is_stable_when_row_order_changes():
    rows = [
        (
            "[국민건강보험] 건강검진 결과가 발급되었습니다 [URL]에서 즉시 확인하세요",
            "phishing",
        ),
        (
            "[국민건강보험] 건강검진 보고서가 발급되었습니다 [URL]에서 지금 확인하세요",
            "phishing",
        ),
        (
            "오늘 저녁 식사는 일곱 시에 시작합니다",
            "normal",
        ),
    ]

    config = TemplateGroupingConfig(
        similarity_threshold=0.70,
        ngram_range=(2, 4),
    )

    original = prepare_template_groups(
        make_dataframe(rows),
        config=config,
    )

    reversed_result = prepare_template_groups(
        make_dataframe(list(reversed(rows))),
        config=config,
    )

    original_mapping = dict(
        zip(
            original["text_norm"],
            original["template_group_id"],
            strict=True,
        )
    )
    reversed_mapping = dict(
        zip(
            reversed_result["text_norm"],
            reversed_result["template_group_id"],
            strict=True,
        )
    )

    assert original_mapping == reversed_mapping


def test_transitively_similar_messages_are_grouped_together(
    monkeypatch,
):
    df = make_dataframe(
        [
            ("메시지 A", "phishing"),
            ("메시지 B", "phishing"),
            ("메시지 C", "phishing"),
        ]
    )
    df = add_text_fingerprints(df)

    monkeypatch.setattr(
        "data_science.SMSModel.template_grouping.similarity._find_similar_pairs",
        lambda texts, config: [
            (0, 1, 0.90),
            (1, 2, 0.91),
        ],
    )

    result = assign_template_groups(df)

    assert result["template_group_id"].nunique() == 1


def test_prepare_template_groups_adds_required_columns():
    df = make_dataframe(
        [
            ("배송 주소 확인 [URL]", "phishing"),
            ("평범한 일상 대화입니다", "normal"),
        ]
    )

    result = prepare_template_groups(df)

    assert "text_fingerprint" in result.columns
    assert "template_group_id" in result.columns
    assert result["text_fingerprint"].notna().all()
    assert result["template_group_id"].notna().all()


def test_prepare_template_groups_handles_empty_dataframe():
    df = pd.DataFrame(
        {
            "text_norm": pd.Series(dtype="string"),
            "label": pd.Series(dtype="string"),
        }
    )

    result = prepare_template_groups(df)

    assert result.empty
    assert "text_fingerprint" in result.columns
    assert "template_group_id" in result.columns


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_grouping_config_rejects_invalid_threshold(
    threshold: float,
):
    with pytest.raises(
        ValueError,
        match="similarity_threshold",
    ):
        TemplateGroupingConfig(
            similarity_threshold=threshold,
        )


@pytest.mark.slow
def test_real_dataset_template_grouping_smoke():

    from data_science.SMSModel.train_sms import DATA_PATH, load_data

    df_pool, _df_holdout = load_data(DATA_PATH)

    assert not df_pool.empty
    assert "text_norm" in df_pool.columns
    assert "text_fingerprint" in df_pool.columns
    assert "template_group_id" in df_pool.columns

    assert df_pool["text_fingerprint"].is_unique
    assert df_pool["template_group_id"].notna().all()

    assert set(df_pool["label"].unique()).issubset({"normal", "phishing"})
