from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.infrastructure.http_retry import _sleep_before_retry


@pytest.mark.asyncio
async def test_sleep_honors_numeric_retry_after() -> None:
    sleep = AsyncMock()

    with patch(
        "app.infrastructure.http_retry.asyncio.sleep",
        sleep,
    ):
        await _sleep_before_retry(
            attempt=0,
            retry_after="7",
        )

    sleep.assert_awaited_once_with(7.0)


@pytest.mark.asyncio
async def test_sleep_honors_http_date_retry_after() -> None:
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=20)
    sleep = AsyncMock()

    with patch(
        "app.infrastructure.http_retry.asyncio.sleep",
        sleep,
    ):
        await _sleep_before_retry(
            attempt=0,
            retry_after=format_datetime(
                retry_at,
                usegmt=True,
            ),
        )

    delay = sleep.await_args.args[0]
    assert 18.0 <= delay <= 20.0


@pytest.mark.asyncio
async def test_sleep_falls_back_for_invalid_retry_after() -> None:
    sleep = AsyncMock()

    with (
        patch(
            "app.infrastructure.http_retry.asyncio.sleep",
            sleep,
        ),
        patch(
            "app.infrastructure.http_retry.random.uniform",
            return_value=0.0,
        ),
    ):
        await _sleep_before_retry(
            attempt=0,
            retry_after="not-a-date",
        )

    sleep.assert_awaited_once_with(0.25)
