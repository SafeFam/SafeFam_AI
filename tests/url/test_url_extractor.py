import unittest
from app.service.url.extractor import extract_urls_from_text

class TestUrlExtractor(unittest.TestCase):

    def test_no_url_text(self):
        """시나리오 1: URL이 아예 없는 순수 대화 텍스트 -> 빈 리스트 반환"""
        text = "안녕하세요! 오늘 점심 뭐 드실래요? 맛있는 거 추천해주세요."
        self.assertEqual(extract_urls_from_text(text), [])

    def test_single_normal_url(self):
        """시나리오 2: 일반적인 URL이 포함된 문장 -> 정상 추출"""
        text = "국민은행 보안 업데이트 링크입니다. https://www.kookminbank.com 확인해보세요."
        expected = ["https://www.kookminbank.com"]
        self.assertEqual(extract_urls_from_text(text), expected)

    def test_multiple_urls(self):
        """시나리오 3: 한 문장에 서로 다른 URL이 2개 이상 포함된 문장 -> 모두 추출"""
        text = "여기 구글 주소 https://google.com 이랑 네이버 주소 http://naver.com 보냅니다."
        expected = ["https://google.com", "http://naver.com"]
        self.assertEqual(extract_urls_from_text(text), expected)

    def test_url_with_trailing_punctuation(self):
        """시나리오 4: 문장 맨 끝에 온점이나 기호와 함께 URL이 위치한 경우 -> 기호 제외하고 깔끔하게 추출 (억까 방지)"""
        text = "아래 단축 링크를 꼭 클릭해주세요: https://bit.ly/3xyz."
        expected = ["https://bit.ly/3xyz"]
        self.assertEqual(extract_urls_from_text(text), expected)

    def test_duplicate_urls(self):
        """시나리오 5: 동일한 URL이 반복되는 문장 -> 중복 제거되어 1개만 반환"""
        text = "급합니다!! https://bit.ly/3xyz 빨리 확인하세요! 다시 보냅니다 https://bit.ly/3xyz"
        expected = ["https://bit.ly/3xyz"]
        self.assertEqual(extract_urls_from_text(text), expected)

if __name__ == '__main__':
    unittest.main()