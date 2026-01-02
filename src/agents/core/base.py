from abc import ABC, abstractmethod
from typing import Optional


from agents.providers.models.base import GenerationBehaviorSettings, Message


class BaseAgent(ABC):
    @abstractmethod
    async def answer_to(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        raise NotImplementedError
