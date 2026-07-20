from abc import ABC, abstractmethod

class BaseSecurityEngine(ABC):
    @abstractmethod
    async def scan_url(self, url: str) -> dict:
        pass