"""
실제 외부 호스트(Google, TinyURL 등)에 라이브로 접속해 URL 트레이서를 검증하는 통합 테스트.
오프라인/CI 환경에서는 아웃바운드 DNS/HTTP에 의존하므로 기본 실행에서는 제외되며
(pytest.ini의 `-m "not integration"`), 필요할 때 `pytest -m integration`으로 명시적으로
실행해야 한다. 결정론적인 단위 동작 검증은 tests/url/test_url_tracer.py가 담당한다.
"""
import pytest
from app.analysis.url.tracker import trace_url


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normal_url_no_redirect_live():
    """일반 URL은 리다이렉트 없이 자기 자신을 반환해야 함"""
    url = "https://www.google.com"
    result = await trace_url(url)
    assert result == url


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tinyurl_resolution_live():
    """tinyurl 단축 주소가 원본 도메인으로 잘 풀리는지 검증"""
    short_url = "https://tinyurl.com/app-store"
    result = await trace_url(short_url)
    assert "tinyurl.com" not in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_broken_url_graceful_handling_live():
    """존재하지 않는 이상한 URL이어도 서버가 안 터지고 본래 값을 뱉어내는지 검증"""
    invalid_url = "https://this-is-completely-broken-domain-12345.com"
    result = await trace_url(invalid_url, timeout=2.0)
    assert result == invalid_url
