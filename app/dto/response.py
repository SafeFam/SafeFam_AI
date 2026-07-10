from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    status: str
    message: str
    data: Optional[T] = None

    @classmethod
    def success(cls, data: T, message: str = "요청이 성공했습니다.") -> "ApiResponse[T]":
        return cls(status="SUCCESS", message=message, data=data)
    
    @classmethod
    def error(cls, message: str) -> "ApiResponse[None]":
        return cls(status="ERROR", message=message, data=None)