import pytest
import asyncio
from app.service.url.tracer import resolve_short_url

@pytest.mark.asyncio
async def test_normal_url_no_redirect():
    """일반 URL은 리다이렉트 없이 자기 자신을 반환해야 함"""
    url = "https://www.google.com"
    result = await resolve_short_url(url)
    assert result == url

@pytest.mark.asyncio
async def test_tinyurl_resolution():
    """tinyurl 단축 주소가 원본 구글 주로소 잘 깨지는지 검증"""
    # 임의로 생성된 구글 리다이렉트 단축 주소 예시
    short_url = "https://tinyurl.com/app-store"
    result = await resolve_short_url(short_url)
    # 종착지가 특정 앱스토어 링크나 정상 도메인으로 복원되는지 확인
    assert "tinyurl.com" not in result

@pytest.mark.asyncio
async def test_broken_url_graceful_handling():
    """존재하지 않는 이상한 URL이어도 서버가 안 터지고 본래 값을 뱉어내는지 검증"""
    invalid_url = "https://this-is-completely-broken-domain-12345.com"
    result = await resolve_short_url(invalid_url, timeout=2.0)
    assert result == invalid_url

@pytest.mark.asyncio
async def test_max_redirect_limit():
    """최대 리다이렉트 제한 횟수가 정상 작동하는지 (1회 제한 테스트)"""
    short_url = "https://tinyurl.com/app-store"
    # max_redirects를 0으로 주면 바로 튕겨서 원래 주소가 나와야 함
    result = await resolve_short_url(short_url, max_redirects=0)
    assert result == short_url