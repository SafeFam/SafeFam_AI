class ProcessingError(RuntimeError):
    """메시지 처리 오류의 공통 기반 클래스"""

    def __init__(
        self,
        message: str,
        failure_code: str,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class RetryableProcessingError(ProcessingError):
    """일시적인 문제로 재시도할 수 있는 오류"""


class NonRetryableProcessingError(ProcessingError):
    """재시도해도 성공할 가능성이 없는 오류"""