"""
Anthropic Provider Module

This module provides integration with Anthropic's Claude language models.
It implements the BaseProvider interface for communication with Anthropic's API,
handling message formatting, tool calling, and response processing specific to Claude models.
The AnthropicProvider class manages the conversion between the system's standardized
message format and Anthropic's API requirements, including special handling for
thinking mode, tool usage, and ephemeral caching controls.
"""

import os
from typing import Any, Optional

import anthropic
from anthropic import NOT_GIVEN

from agents.config import LLM_HTTP_TIMEOUT
from agents.config.models import model_registry
from agents.providers.models.base import (
    BaseProvider,
    GenerationBehaviorSettings,
    History,
    LLMThought,
    Message,
    RoleEnum,
    ToolCallRequest,
)
from agents.providers.models.exceptions import (
    LLMContextOverflowError,
    ProviderFailureError,
)
from agents.providers.models.token_usage import BaseUsageLogEntry
from agents.utils.logs.config import logger

DEFAULT_ANTHROPIC_GBS = GenerationBehaviorSettings(temperature=1)


class AnthropicProvider(BaseProvider):
    THOUGHT_TYPE_REDACTED = "claude_thought_redacted"
    THOUGH_TYPE_THINKING = "claude_thought"
    TIMEOUT_RETRIES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_signatures = self._get_signatures()
        self.model_config = model_registry.get_model("anthropic", self.version) or model_registry.find_model_by_alias(
            self.version
        )

    def _get_model_config(self):
        return model_registry.get_model("anthropic", self.version) or model_registry.find_model_by_alias(self.version)

    def map_role(self, role: RoleEnum) -> str:
        return {
            RoleEnum.SYSTEM: "user",
            RoleEnum.ASSISTANT: "assistant",
            RoleEnum.USER: "user",
            RoleEnum.TOOL: "user",
            RoleEnum.DEVELOPER: "user",
        }[role]

    def _prepare_messages(self, history: History) -> list[dict[str, str]]:
        messages = []
        for i, message in enumerate(history.messages):
            if message.thoughts:
                for thought in message.thoughts:
                    if thought.type == self.THOUGHT_TYPE_REDACTED:
                        content = {
                            "data": thought.content,
                            "type": "redacted_thinking",
                        }
                    elif thought.type == self.THOUGH_TYPE_THINKING and thought.signature is not None:
                        content = {
                            "signature": thought.signature,
                            "thinking": thought.content,
                            "type": "thinking",
                        }
                    else:
                        continue

                    messages.append(
                        {
                            "role": self.map_role(RoleEnum.ASSISTANT),
                            "content": [content],
                        }
                    )

            if message.message_type == "text":
                data = {
                    "role": self.map_role(message.role),
                    "content": message.content,
                }
            elif message.message_type == "tool_call":
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for tool_call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.tool_call_id,
                            "name": tool_call.tool_name,
                            "input": tool_call.tool_args,
                        }
                    )
                data = {
                    "role": self.map_role(message.role),
                    "content": content,
                }

            elif message.message_type == "tool_result":
                contents = []
                for tool_result in message.tool_call_results:
                    content = {
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": tool_result.content,
                    }
                    contents.append(content)

                data = {
                    "role": self.map_role(message.role),
                    "content": contents,
                }

            if i == len(history.messages) - 1:
                if isinstance(data["content"], str):
                    data["content"] = [
                        {
                            "type": "text",
                            "text": data["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(data["content"], list) and data["content"]:
                    data["content"][-1]["cache_control"] = {"type": "ephemeral"}

            messages.append(data)

        return messages

    def _prepare_completions_arguments(self, history: History, gbs: GenerationBehaviorSettings) -> dict[str, Any]:
        messages = []
        for message in self._prepare_messages(history):
            messages.append(message)

        tools = self.tool_signatures if self.tools else []
        max_tokens = gbs.max_output_tokens if gbs.max_output_tokens else self.model_config.token_config.output_tokens
        extra_headers = None
        thinking = {"type": "disabled"}
        temperature = NOT_GIVEN if gbs.temperature is None else gbs.temperature

        if gbs.thinking:
            thinking = {"type": "enabled", "budget_tokens": gbs.thinking_budget}
            temperature = 1
            if gbs.options.get("interleaved_thinking", False):
                extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        if gbs.web_search:
            tools.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": gbs.max_built_in_tool_calls,
                }
            )
        return {
            "model": self.version,
            "max_tokens": max_tokens,
            "extra_headers": extra_headers,
            "thinking": thinking,
            "temperature": temperature,
            "tools": tools,
            "messages": messages,
            "system": [
                {
                    "type": "text",
                    "text": self.full_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "metadata": {"user_id": gbs.client_identifier} if gbs.client_identifier else NOT_GIVEN,
        }

    async def _llm(self, history, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        gbs = gbs or DEFAULT_ANTHROPIC_GBS
        error = None
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=LLM_HTTP_TIMEOUT * 1000)
        completions_kwargs = self._prepare_completions_arguments(history, gbs)
        for _ in range(self.TIMEOUT_RETRIES):
            try:
                response = await client.messages.create(**completions_kwargs)
                self._new_usage(response.usage)
                return await self._response_to_message(response, history, gbs)
            except (
                anthropic.APITimeoutError,
                anthropic.APIError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
            ) as e:
                logger.exception(f"Anthropic API error (attempt {_ + 1}/3): {str(e)}")
                if e.status_code == 400 and "prompt is too long" in e.body.get("error", {}).get("message"):
                    raise LLMContextOverflowError() from e
                error = e

        raise ProviderFailureError("Anthropic provider failed after 3 retries") from error

    async def _response_to_message(
        self,
        response,
        history: History,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ) -> Message:
        messages, tool_calls, thoughts = [], [], []
        for content in response.content:
            if content.type == "text":
                messages.append(content.text)

            elif content.type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        tool_call_id=content.id,
                        tool_name=content.name,
                        tool_args=content.input,
                    )
                )
            elif content.type == "server_tool_use":
                logger.info("Anthropic: Server tool use: " + content.model_extra["name"])
            elif content.type == "web_search_tool_result":
                pass
            elif content.type == "thinking":
                thoughts.append(
                    LLMThought(
                        content=content.thinking,
                        signature=content.signature,
                        type=self.THOUGH_TYPE_THINKING,
                    )
                )
            elif content.type == "redacted_thinking":
                thoughts.append(
                    LLMThought(
                        content=content.redacted_thinking,
                        signature=content.signature,
                        type=self.THOUGHT_TYPE_REDACTED,
                    )
                )
            else:
                raise Exception("Not known content type")

        return Message(
            message_type="text" if not tool_calls else "tool_call",
            role=RoleEnum.ASSISTANT,
            content="".join(messages),
            tool_calls=tool_calls,
            thoughts=thoughts,
        )

    def _arg_to_schema(self, arg_schema: dict | None, description: str | None = None) -> dict:
        if not arg_schema:
            return {"type": "string", "description": description}

        schema = {
            "type": arg_schema.get("type", "string"),
        }

        if arg_schema.get("description"):
            schema["description"] = arg_schema.get("description", "")

        if arg_schema.get("items"):
            schema["items"] = self._arg_to_schema(arg_schema["items"])

        if arg_schema.get("properties"):
            schema["properties"] = {
                field: self._arg_to_schema(field_schema) for field, field_schema in arg_schema["properties"].items()
            }

        return schema

    def _get_signatures(self) -> list[dict]:
        signatures = []
        for tool in self.tools:
            name = tool.name
            properties = {}
            for arg_name, description in tool.args_description.items():
                properties[arg_name] = self._arg_to_schema(tool.args_schema.get(arg_name, None), description)

            signature = {
                "name": name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool.args_description.keys()),
                },
            }
            signatures.append(signature)

        if signatures:
            signatures[-1]["cache_control"] = {"type": "ephemeral"}

        return signatures

    def _new_usage(self, usage):
        try:
            self.add_usage(
                AnthropicUsageLogEntry(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_write_input_tokens=usage.cache_creation_input_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    model_version=self.version,
                )
            )
        except Exception:
            logger.exception(f"Failed to log usage for Anthropic model: {usage}.")

    async def count_tokens(self, history: History, gbs: Optional[GenerationBehaviorSettings] = None) -> int:
        gbs = gbs or DEFAULT_ANTHROPIC_GBS
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=LLM_HTTP_TIMEOUT * 1000)
        completions_kwargs = self._prepare_completions_arguments(history, gbs)
        valid_args = [
            "messages",
            "model",
            "system",
            "thinking",
            "tool_choice",
            "tools",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]
        kwargs = {}

        for key, value in completions_kwargs.items():
            if key in valid_args:
                kwargs[key] = value

        response = await client.messages.count_tokens(**kwargs)
        return response.input_tokens


class AnthropicUsageLogEntry(BaseUsageLogEntry):
    model_version: str
    input_tokens: int
    cache_write_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int

    def calculate_cost(self) -> float:
        model_config = model_registry.get_model("anthropic", self.model_version) or model_registry.find_model_by_alias(
            self.model_version
        )

        if not model_config:
            logger.warning(f"No pricing configuration found for model version {self.model_version}.")
            return 0.0

        pricing = model_config.pricing
        input_tokens_cost = self.input_tokens * pricing.input_per_1m
        cached_write_tokens_cost = self.cache_write_input_tokens * (pricing.cache_creation_per_1m or 0)
        cached_read_tokens_cost = self.cache_read_input_tokens * (pricing.cache_read_per_1m or 0)
        output_tokens_cost = self.output_tokens * pricing.output_per_1m
        return (input_tokens_cost + cached_write_tokens_cost + cached_read_tokens_cost + output_tokens_cost) / 1_000_000
