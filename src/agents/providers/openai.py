"""
OpenAI Provider Module

This module provides integration with OpenAI's language models using the Responses API exclusively.
It implements the BaseProvider interface for communication with OpenAI's Responses API,
handling message formatting, tool calling, web search, and response processing for all OpenAI models.
The OpenAIProvider class converts conversation history to input text for the Responses API,
with unified handling for all model versions (o1, o3, o4-mini, gpt-5, gpt-4o, etc.).
"""

import json
import os
from typing import Any, Dict, List, Optional

import openai
from openai import NOT_GIVEN, AsyncOpenAI

from agents.config import LLM_HTTP_TIMEOUT
from agents.config.models import model_registry
from agents.providers.models.base import (
    BaseProvider,
    GenerationBehaviorSettings,
    LLMThought,
    Message,
    RoleEnum,
    ToolCallRequest,
)
from agents.providers.models.exceptions import LLMContextOverflowError, ProviderFailureError
from agents.providers.models.token_usage import BaseUsageLogEntry
from agents.utils.logs.config import logger

DEFAULT_OPENAI_GBS = GenerationBehaviorSettings(temperature=None)


class OpenAIProvider(BaseProvider):
    THOUGH_TYPE_THINKING = "openai_thought"
    TIMEOUT_RETRIES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_signatures = self._get_signatures()
        self.model_config = model_registry.get_model("openai", self.version) or model_registry.find_model_by_alias(
            self.version
        )

    def map_role(self, role: RoleEnum) -> str:
        return {
            RoleEnum.SYSTEM: "system",
            RoleEnum.ASSISTANT: "assistant",
            RoleEnum.USER: "user",
            RoleEnum.TOOL: "tool",
            RoleEnum.DEVELOPER: "developer",
        }[role]

    def _get_model_config(self):
        return model_registry.get_model("openai", self.version) or model_registry.find_model_by_alias(self.version)

    def _get_web_search_config_from_gbs(self, gbs: GenerationBehaviorSettings) -> Dict[str, Any]:
        """Extract web search configuration from GenerationBehaviorSettings options."""
        config = {}

        if gbs.options.get("web_search_allowed_domains"):
            config["allowed_domains"] = gbs.options["web_search_allowed_domains"]

        if gbs.options.get("web_search_user_location"):
            config["user_location"] = {"type": "approximate", **gbs.options["web_search_user_location"]}
        return config

    def _prepare_messages(self, history) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": self.full_prompt,
            }
        ]
        for message in history.messages:
            if message.message_type == "text":
                messages.append(
                    {
                        "role": self.map_role(message.role),
                        "content": message.content,
                    }
                )
            elif message.message_type == "tool_call":
                # Add assistant message with tool calls
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.tool_args),
                            },
                        }
                    )
                messages.append({"role": "assistant", "content": message.content or None, "tool_calls": tool_calls})
            elif message.message_type == "tool_result":
                # Add tool response messages
                for tool_call_result in message.tool_call_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_result.tool_call_id,
                            "content": tool_call_result.content,
                        }
                    )

        return messages

    async def _llm(self, history, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        gbs = gbs or DEFAULT_OPENAI_GBS
        error = None

        tool_signatures = self._get_signatures() if self.tools else None

        async with AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=LLM_HTTP_TIMEOUT,
        ) as client:
            for _ in range(self.TIMEOUT_RETRIES):
                try:
                    response_params = {
                        "model": self.version,
                        "messages": self._prepare_messages(history),
                        "max_tokens": gbs.max_output_tokens if gbs.max_output_tokens else NOT_GIVEN,
                    }

                    if tool_signatures:
                        response_params["tools"] = tool_signatures
                        response_params["tool_choice"] = "auto"

                    if gbs.temperature is not None:
                        response_params["temperature"] = gbs.temperature

                    response = await client.chat.completions.create(**response_params)

                    if hasattr(response, "usage"):
                        self._new_usage(response.usage)
                    return self._response_to_message_chat(response)

                except (
                    openai.APITimeoutError,
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                ) as exc:
                    if hasattr(exc, "status_code") and exc.status_code == 400:
                        if (
                            hasattr(exc, "body")
                            and isinstance(exc.body, dict)
                            and exc.body.get("code") == "context_length_exceeded"
                        ):
                            raise LLMContextOverflowError() from exc
                    error = exc

        raise ProviderFailureError(f"OpenAI provider failed after {self.TIMEOUT_RETRIES} retries") from error

    def _response_to_message_chat(self, response):
        """Convert chat completion response to Message"""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    ToolCallRequest(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        tool_args=json.loads(tool_call.function.arguments),
                    )
                )

        return Message(
            message_type="text" if not tool_calls else "tool_call",
            role=RoleEnum.ASSISTANT,
            content=message.content or "",
            tool_calls=tool_calls,
            thoughts=[],
        )

    def _response_to_message(self, response, tool_signatures: Optional[List[dict]] = None):
        contents = []
        reasonings = []
        tool_calls = []

        has_output_text = hasattr(response, "output_text") and response.output_text
        if has_output_text:
            contents.append(response.output_text)

        for item in response.output:
            if item.type == "message" and not has_output_text:
                for content_item in item.content:
                    if content_item.type == "output_text":
                        content_text = content_item.text
                        contents.append(content_text)

            elif item.type == "reasoning":
                reasonings.append(
                    LLMThought(
                        id=item.id,
                        content=None,
                        signature=item.encrypted_content,
                        type=self.THOUGH_TYPE_THINKING,
                    )
                )
            elif item.type == "function_call" and tool_signatures:
                arguments = item.arguments
                arguments = json.loads(arguments)
                tool_calls.append(
                    ToolCallRequest(
                        tool_call_id=item.id,
                        tool_name=item.name,
                        tool_args=arguments,
                    )
                )

        return Message(
            message_type="text" if not tool_calls else "tool_call",
            role=RoleEnum.ASSISTANT,
            content="\n".join(contents),
            tool_calls=tool_calls,
            thoughts=reasonings,
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
        """Generate tool signatures for chat completions API"""
        signatures = []

        for tool in self.tools:
            name = tool.name
            properties = {}
            for arg_name, description in tool.args_description.items():
                properties[arg_name] = self._arg_to_schema(tool.args_schema.get(arg_name, None), description)

            signature = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(tool.args_description.keys()),
                    },
                },
            }

            signatures.append(signature)
        return signatures

    def _new_usage(self, usage):
        try:
            # Handle both CompletionUsage (chat completions) and ResponseUsage (responses API)
            input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
            cached_tokens = 0

            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
            elif hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0)

            self.add_usage(
                OpenAIUsageLogEntry(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    model_version=self.version,
                )
            )
        except Exception:
            logger.exception(f"Failed to log usage for OpenAIProvider: {usage}")


class OpenAIUsageLogEntry(BaseUsageLogEntry):
    model_version: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int

    def calculate_cost(self) -> float:
        model_config = model_registry.get_model("openai", self.model_version) or model_registry.find_model_by_alias(
            self.model_version
        )

        if not model_config:
            logger.warning(f"No pricing configuration found for model version {self.model_version}.")
            return 0.0

        pricing = model_config.pricing
        input_tokens_cost = self.input_tokens * pricing.input_per_1m
        cached_tokens_cost = self.cached_tokens * (pricing.cached_input_per_1m or 0)
        output_tokens_cost = self.output_tokens * pricing.output_per_1m
        return (input_tokens_cost + cached_tokens_cost + output_tokens_cost) / 1_000_000
