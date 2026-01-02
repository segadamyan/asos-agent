from abc import ABCMeta, abstractmethod
from typing import List

from pydantic import BaseModel


class BaseUsageLogEntry(BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def calculate_cost(self) -> float:
        """Calculate the cost of this usage entry."""
        raise NotImplementedError("Subclasses must implement this method.")


class TokenUsageLog:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(TokenUsageLog, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self._usage: List[BaseUsageLogEntry] = self._usage if hasattr(self, "_usage") else []

    def add(self, usage_entry: BaseUsageLogEntry):
        """Add a new usage entry to the log."""
        self._usage.append(usage_entry)

    def get_total_cost(self) -> float:
        """Calculate the total cost of all usage entries in USD."""
        return sum(entry.calculate_cost() for entry in self._usage)
