import re

# URL을 탐지하기 위한 정규표현식 패턴
URL_PATTERN = re.compile(r'https?://[^\s\'"<>]+')

def extract_urls_from_text(text: str) -> list[str]:
    """
    텍스트 본문에서 모든 웹 URL 주소를 추출하고 중복을 제거하여 반환합니다.
    """

    if not text:
        return []
    
    # 정규식 매칭
    raw_urls = URL_PATTERN.findall(text)

    # 문장 끝에 붙은 불필요한 문장부호 우측 정제
    cleaned_urls = []
    for url in raw_urls:
        cleaned_urls = url.rstrip('.,?!:;)[]')
        cleaned_urls.append(cleaned_urls)

    # 추출 순서를 보존하며 중복 제거
    unique_urls = list(dict.fromkeys(cleaned_urls))

    return unique_urls