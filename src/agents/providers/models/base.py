"""
Provider Base Module

This module defines the foundational classes and interfaces for AI model providers.
It contains base implementations for message handling, tool calling, context window
management, and LLM provider interactions. Core components include message structure
definitions, history management, generation behavior settings, and pricing tables
for various AI models. This module serves as the backbone for all provider-specific
implementations in the system.
"""

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.providers.models.token_usage import BaseUsageLogEntry, TokenUsageLog
from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger


class ReasoningEffort(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RoleEnum(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    DEVELOPER = "developer"


class ToolCallRequest(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    metadata: Optional[dict[str, Any]] = {}

    def __str__(self):
        return f"Request to call a tool `{self.tool_name}` with args: {self.tool_args}"


class LLMThought(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    signature: Optional[str] = None
    type: str


class ToolCallResult(BaseModel):
    type: str = "tool_result"
    success: bool = True
    tool_name: str
    tool_call_id: str
    content: str
    tool_args: dict[str, Any]


class Message(BaseModel):
    message_type: Literal["text", "tool_call", "tool_result", "retry"] = "text"
    role: RoleEnum
    content: str = ""
    additional_message_type: Optional[Literal["system_message", "observation", "ephemeral"]] = None
    tool_calls: List[ToolCallRequest] = []
    tool_call_results: List[ToolCallResult] = []
    thoughts: List[LLMThought] = []


class BaseContextWindowManager(metaclass=ABCMeta):
    @abstractmethod
    async def compact(self, messages: list[Message]) -> list[Message]:
        raise NotImplementedError


class History(BaseModel):
    messages: list[Message] = []
    compaction_on: Optional[bool] = False
    _all_messages = []

    def clear(self):
        self.messages.clear()

    def add_user_message(self, content: str):
        self.add_message(Message(role=RoleEnum.USER, content=content))

    def add_assistant_message(self, content: str):
        self.add_message(Message(role=RoleEnum.ASSISTANT, content=content))

    def add_developer_message(self, content: str):
        self.add_message(Message(role=RoleEnum.DEVELOPER, content=content))

    def add_system_message(self, content: str, additional_message_type: Optional[str] = None):
        self.add_message(
            Message(
                role=RoleEnum.SYSTEM,
                content=content,
                additional_message_type=additional_message_type,
            )
        )

    def add_message(self, message: Message):
        self._all_messages.append(message)
        self.messages.append(message)

    async def compact(self, context_window_manager: BaseContextWindowManager):
        if not self.compaction_on:
            raise ValueError("Compaction is not enabled, can't do compaction.")

        newly_compacted = await context_window_manager.compact(self.messages)
        self.messages = newly_compacted


class GenerationBehaviorSettings(BaseModel):
    """Configuration for controlling response generation behavior."""

    temperature: Optional[float] = Field(default=None, description="Controls randomness in generation (0.0-1.0)")
    thinking: bool = Field(default=False, description="Whether to use thinking for responses")
    thinking_budget: int = Field(default=16000, description="Budget for thinking")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum number of output tokens allowed")
    url_context_analysis: bool = Field(default=False, description="Whether to use context analysis for responses")
    web_search: bool = Field(default=False, description="Whether to enable web search capabilities")
    max_built_in_tool_calls: int = Field(default=10, description="Maximum number of built-in tool calls allowed")
    client_identifier: Optional[str] = Field(
        default=None,
        description="Client identifier which could be used by providers to identify which client/user "
        "is responsible for abuse or violations.",
    )
    options: dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific options")

    @field_validator("temperature", mode="before")
    def validate_temperature(cls, value):
        if (value is not None) and not (0 <= value <= 1):
            raise ValueError("Temperature must be between 0 and 1")
        return value

    @field_validator("options", mode="before")
    def validate_options(cls, value):
        """Validate web search options within the options dict."""
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise ValueError("Options must be a dictionary")

        if "web_search_allowed_domains" in value:
            domains = value["web_search_allowed_domains"]
            if domains is not None:
                if not isinstance(domains, list):
                    raise ValueError("Web search allowed domains must be a list")
                if len(domains) > 20:
                    raise ValueError("Web search allowed domains cannot exceed 20 domains")
                cleaned_domains = []
                for domain in domains:
                    if isinstance(domain, str):
                        cleaned_domain = domain.replace("https://", "").replace("http://", "")
                        cleaned_domains.append(cleaned_domain)
                    else:
                        raise ValueError("All domains in web_search_allowed_domains must be strings")
                value["web_search_allowed_domains"] = cleaned_domains

        if "web_search_user_location" in value:
            location = value["web_search_user_location"]
            if location is not None:
                if isinstance(location, dict):
                    if "country" in location and location["country"] is not None:
                        if len(location["country"]) != 2:
                            raise ValueError("Country code should be a 2-letter ISO code")
                else:
                    raise ValueError("Web search user location must be a dictionary")

        if "reasoning_effort" in value:
            effort = value["reasoning_effort"]
            if effort is not None:
                if not isinstance(effort, str):
                    raise ValueError("Reasoning effort must be a string")
                if effort not in [e.value for e in ReasoningEffort]:
                    raise ValueError(f"Reasoning effort must be one of: {[e.value for e in ReasoningEffort]}")

        return value


class BaseProvider(metaclass=ABCMeta):
    MAX_RETRIES = 3

    __usage_log = TokenUsageLog()

    def __init__(
        self,
        prompt: str,
        version: str = None,
        tools: list[ToolDefinition] = None,
        context_window_manager: Optional[BaseContextWindowManager] = None,
        *args,
        **kwargs,
    ):
        self.raw_prompt = prompt
        self.version = version
        self.tools = [] if tools is None else tools
        self.context_window_manager = context_window_manager
        self.model_config = self._get_model_config()

    @abstractmethod
    def map_role(self, role: RoleEnum) -> str:
        raise NotImplementedError

    @abstractmethod
    async def _llm(self, history, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        raise NotImplementedError

    @abstractmethod
    def _get_model_config(self):
        raise NotImplementedError

    async def get_response(
        self,
        history: Optional[History] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ) -> Message:
        if not history:
            history = History()

        if history.compaction_on:
            if self.context_window_manager is None:
                raise ValueError(f"Compaction is enabled but no LLM context manager is provided to {self.__class__}.")
            await history.compact(self.context_window_manager)

        message = await self._llm(history, gbs)
        return message

    @property
    def full_prompt(self) -> str:
        """Combines base prompt with tool descriptions."""
        return self.raw_prompt

    def add_usage(self, usage_entry: BaseUsageLogEntry):
        """Add a usage entry to the provider's usage log."""
        self.__usage_log.add(usage_entry)
        logger.info(f"Total-Usage-Cost: ${self.__usage_log.get_total_cost()}")

    def has_thinking_capabilities(self) -> bool:
        """Check if this provider has thinking capabilities."""
        if self.model_config is None:
            raise ValueError(
                f"Model configuration is not available for provider {self.__class__.__name__} "
                f"with version '{self.version}'. Cannot determine thinking capabilities. "
                f"Please ensure the model is properly configured in models.yaml."
            )
        return "thinking" in self.model_config.capabilities


class ProvidersEnum(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"


class IntelligenceProviderConfig(BaseModel):
    provider_name: ProvidersEnum
    version: str

    model_config = ConfigDict(extra="forbid")
