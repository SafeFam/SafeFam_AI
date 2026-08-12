from unittest.mock import patch

import pytest

from app.infrastructure.llm.factory import (
    get_llm_client,
    reset_llm_client_for_test,
)


@pytest.fixture(autouse=True)
def reset_factory_cache():
    reset_llm_client_for_test()
    yield
    reset_llm_client_for_test()


def test_factory_builds_and_caches_bedrock_client():
    sentinel = object()

    with patch(
        "app.infrastructure.llm.factory.BedrockLlmClient",
        return_value=sentinel,
    ) as constructor:
        first = get_llm_client()
        second = get_llm_client()

    assert first is sentinel
    assert second is sentinel
    constructor.assert_called_once_with()


def test_reset_llm_client_clears_cached_instance():
    with patch("app.infrastructure.llm.factory.BedrockLlmClient") as constructor:
        constructor.side_effect = [object(), object()]
        first = get_llm_client()
        reset_llm_client_for_test()
        second = get_llm_client()

    assert first is not second
    assert constructor.call_count == 2


def test_factory_rejects_unsupported_provider():
    with patch("app.infrastructure.llm.factory.settings.LLM_PROVIDER", "other"):
        with pytest.raises(ValueError, match="unsupported LLM provider: other"):
            get_llm_client()
