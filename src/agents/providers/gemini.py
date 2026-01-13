"""
Gemini Provider Module

This module provides integration with Google's Gemini language models.
It implements the BaseProvider interface for communication with Google's Generative AI API,
handling message formatting, tool calling, response processing, and grounding capabilities
specific to Gemini models. The GeminiProvider class manages the conversion between the
system's standardized message format and Google's API requirements, with robust error
handling, retries for transient failures, and special processing for search grounding
and function calling features.
"""

import asyncio
import base64
import os
import uuid
from typing import Optional

from aiohttp import ServerDisconnectedError
from google import genai
from google.genai import Client, types
from google.genai.errors import ClientError, ServerError
from google.genai.types import FinishReason
from requests.exceptions import ReadTimeout

from agents import config
from agents.config.models import model_registry
from agents.providers.models.base import (
    BaseProvider,
    GenerationBehaviorSettings,
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

DEFAULT_GEMINI_GBS = GenerationBehaviorSettings(temperature=1)


class GeminiProvider(BaseProvider):
    THOUGH_TYPE_THINKING = "gemini_thought"

    def __init__(self, *args, **kwargs):  # noqa
        super().__init__(*args, **kwargs)
        self.tool_signatures = self._get_signatures()
        self.model_config = model_registry.get_model("gemini", self.version) or model_registry.find_model_by_alias(
            self.version
        )

    def map_role(self, role: RoleEnum) -> str:
        return {
            RoleEnum.SYSTEM: "user",
            RoleEnum.ASSISTANT: "model",
            RoleEnum.USER: "user",
            RoleEnum.TOOL: "user",
            RoleEnum.DEVELOPER: "user",
        }[role]

    def _get_model_config(self):
        return model_registry.get_model("gemini", self.version) or model_registry.find_model_by_alias(self.version)

    def _prepare_messages(self, history) -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": self.raw_prompt,
                    }
                ],
            }
        ]
        for message in history.messages:
            for thought in message.thoughts:
                if thought.type == self.THOUGH_TYPE_THINKING or thought.signature is None:
                    messages.append(
                        {
                            "role": "model",
                            "parts": [
                                {
                                    "thought": True,
                                    "thought_signature": thought.signature,
                                    "text": thought.content,
                                }
                            ],
                        }
                    )

            if message.message_type == "text":
                messages.append(
                    {
                        "role": self.map_role(message.role),
                        "parts": [
                            {
                                "text": message.content,
                            }
                        ],
                    }
                )
            elif message.message_type == "tool_call":
                parts = []
                if message.content:
                    parts.append({"text": message.content})
                for tool_call in message.tool_calls:
                    part = {
                        "function_call": {
                            "name": tool_call.tool_name,
                            "args": tool_call.tool_args,
                        }
                    }
                    if "thought_signature" in tool_call.metadata:
                        signature = tool_call.metadata["thought_signature"]
                        if signature is not None:
                            signature = base64.b64decode(signature)
                        part["thought_signature"] = signature
                    parts.append(part)
                messages.append(
                    {
                        "role": self.map_role(message.role),
                        "parts": parts,
                    }
                )
            elif message.message_type == "tool_result":
                parts = []
                for tool_result in message.tool_call_results:
                    parts.append(
                        {
                            "function_response": {
                                "id": tool_result.tool_call_id,
                                "name": tool_result.tool_name,
                                "response": {
                                    "output": tool_result.content,
                                },
                            }
                        }
                    )
                messages.append(
                    {
                        "role": self.map_role(message.role),
                        "parts": parts,
                    }
                )
        return messages

    async def _llm(self, history, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        gbs = gbs or DEFAULT_GEMINI_GBS
        messages = self._prepare_messages(history)
        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options={
                "timeout": config.LLM_HTTP_TIMEOUT * 1000,
            },
        )

        tools = [types.Tool(function_declarations=[signature]) for signature in self.tool_signatures]
        genai_config = {
            "tools": tools,
            "automatic_function_calling": {"disable": True},
            "temperature": gbs.temperature,
            "max_output_tokens": gbs.max_output_tokens,  # None is a valid value; the API will handle it if not set
        }

        if self.has_thinking_capabilities():
            logger.warning("Enabling thinking capability for Gemini model.")
            gbs = gbs.model_copy()
            gbs.thinking = True
            genai_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=gbs.thinking_budget, include_thoughts=True
            )
        else:
            genai_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=gbs.thinking_budget if gbs.thinking else 0,
                include_thoughts=gbs.thinking,
            )

        if tools:
            genai_config["tool_config"] = {"function_calling_config": {"mode": "auto"}}

        if gbs.web_search:
            tools.append(
                types.Tool(
                    google_search=types.GoogleSearch(),
                )
            )

        if gbs.url_context_analysis:
            tools.append(
                types.Tool(
                    url_context=types.UrlContext(),
                )
            )

        response = await self._request(client, genai_config, messages)
        return self._response_to_message(response)

    async def _request(self, client: Client, genai_config: dict, messages: list, retries: int = 10):
        """
        Make a request to the model with the given messages and configuration.
        """
        failed_finish_reasons = (
            FinishReason.MALFORMED_FUNCTION_CALL.name,
            FinishReason.RECITATION.name,
        )
        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=self.version, config=genai_config, contents=messages
                )
                if not response:
                    logger.warning("Empty response received from Gemini API. Retrying...")
                    continue

                if hasattr(response, "candidates") and not response.candidates:
                    logger.warning("No candidates found in the response. Retrying...")
                    continue

                if any(candidate.finish_reason.name in failed_finish_reasons for candidate in response.candidates):
                    logger.warning("Function call with large arguments detected. Retrying with reduced size.")
                    messages.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": "You are making function call with large arguments. Please immediately "
                                    "recall the same action but reduce the size of the arguments.",
                                }
                            ],
                        }
                    )
                    continue
                self._new_usage(response.usage_metadata)
                return response

            except ClientError as e:
                logger.exception(
                    "ClientError: The prompt was blocked for safety reasons. "
                    f"Please review and modify your prompt to comply with safety guidelines. Error: {e}",
                )
                if e.code == 429:
                    logger.warning("ClientError: Rate limit error received, retrying...")
                    if self._context_overflow(e):
                        raise LLMContextOverflowError() from e
                    await asyncio.sleep(2**attempt)
                elif e.code == 400:
                    if "exceeds the maximum number of tokens allowed" in str(e):
                        raise LLMContextOverflowError() from e
                logger.warning(
                    f"ClientError: Received client error (code {e.code}) on attempt {attempt + 1} of {retries}. "
                    f"This may be a client side issue. Error: {e}",
                )
            except ServerError as e:
                # Server errors indicate issues on the API provider side.
                logger.warning(
                    f"ServerError: Received server error (code {e.code}) on attempt {attempt + 1} of {retries}. "
                    f"This may be a temporary server-side issue. Error: {e}",
                )
                if e.code < 500 and e.code != 429:
                    # Client-side error returned as ServerError: do not retry.
                    logger.error(
                        f"ServerError: Error code {e.code} indicates a non-transient error. Aborting retries.",
                    )
                    raise ProviderFailureError("Gemini provider failed with non-transient error") from e
                await asyncio.sleep(2**attempt)
            except ReadTimeout:
                logger.warning("Read timeout error: Retrying...")
                await asyncio.sleep(2**attempt)
            except ServerDisconnectedError:
                logger.warning("Server Disconnected error: Retrying...")
                await asyncio.sleep(2**attempt)

        raise ProviderFailureError("Gemini provider failed after max retries") from e

    def _response_to_message(self, response):
        """
        Parse the response and extract text, code, and grounding information.

        Args:
            response: Raw response from the model

        Returns:
            Dictionary containing parsed content and grounding information
        """
        messages, tool_calls, groundings, thoughts = [], [], [], []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Extract text and code from parts
            if hasattr(candidate.content, "parts"):
                if not candidate.content.parts:
                    print(candidate.finish_reason.name)
                    # import pdb ; pdb.set_trace()

                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thoughts.append(
                            LLMThought(
                                content=part.text,
                                signature=part.thought_signature,
                                type=self.THOUGH_TYPE_THINKING,
                            )
                        )
                    else:
                        if hasattr(part, "text") and part.text:
                            messages.append(part.text)
                        if hasattr(part, "function_call") and part.function_call:
                            metadata = {}
                            if hasattr(part, "thought_signature"):
                                if part.thought_signature is None:
                                    metadata["thought_signature"] = None
                                else:
                                    metadata["thought_signature"] = base64.b64encode(part.thought_signature).decode()
                            tool_calls.append(
                                ToolCallRequest(
                                    tool_call_id=uuid.uuid4().hex,
                                    tool_name=part.function_call.name,
                                    tool_args=part.function_call.args,
                                    metadata=metadata,
                                )
                            )

        return Message(
            message_type="text" if not tool_calls else "tool_call",
            role=RoleEnum.ASSISTANT,
            content="\n".join(messages),
            tool_calls=tool_calls,
            thoughts=thoughts,
        )

    def _map_argument_type(self, arg_type: str | None) -> str:
        type_mapping = {
            None: types.Type.TYPE_UNSPECIFIED,
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
        }
        return type_mapping.get(arg_type.lower(), "string")

    def _arg_to_schema(self, arg_schema: dict | None, description: str | None = None) -> types.Schema:
        if not arg_schema:
            return types.Schema(type=types.Type.STRING, description=description)

        return types.Schema(
            type=self._map_argument_type(arg_schema.get("type", "string")),
            description=arg_schema.get("description", ""),
            items=self._arg_to_schema(arg_schema["items"]) if arg_schema.get("items") else None,
            properties={
                field: self._arg_to_schema(field_schema) for field, field_schema in arg_schema["properties"].items()
            }
            if arg_schema.get("properties")
            else None,
        )

    def _get_signatures(self) -> list[types.FunctionDeclaration]:
        signatures = []

        for tool in self.tools:
            properties = {}
            for name, description in tool.args_description.items():
                properties[name] = self._arg_to_schema(tool.args_schema.get(name, None), description)

            parameters = None
            if properties:
                parameters = types.Schema(
                    properties=properties,
                    type="OBJECT",
                )
            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
            signatures.append(function_declaration)
        return signatures

    def _new_usage(self, usage_metadata):
        try:
            self.add_usage(
                GeminiUsageLogEntry(
                    input_tokens=0 if usage_metadata.prompt_token_count is None else usage_metadata.prompt_token_count,
                    output_tokens=0
                    if usage_metadata.candidates_token_count is None
                    else usage_metadata.candidates_token_count,
                    model_version=self.version,
                )
            )
        except Exception:
            logger.exception(f"Failed to log usage for Gemini model {usage_metadata}.")

    @staticmethod
    def _context_overflow(e: ClientError) -> bool:
        if "error" not in e.details:
            return False
        elif "details" not in e.details["error"]:
            return False

        for detail in e.details["error"]["details"]:
            if "violations" not in detail:
                continue

            for violation in detail["violations"]:
                if (
                    "quotaId" in violation
                    and violation["quotaId"] == "GenerateContentPaidTierInputTokensPerModelPerMinute"
                ):
                    return True
        return False


class GeminiUsageLogEntry(BaseUsageLogEntry):
    model_version: str
    input_tokens: int
    output_tokens: int

    def calculate_cost(self) -> float:
        model_config = model_registry.get_model("gemini", self.model_version) or model_registry.find_model_by_alias(
            self.model_version
        )

        if not model_config:
            logger.warning(f"No pricing configuration found for model version {self.model_version}.")
            return 0.0

        pricing = model_config.pricing

        if pricing.threshold and self.input_tokens > pricing.threshold:
            # long context pricing
            input_price_per_1M = pricing.input_per_1m_after or pricing.input_per_1m
            output_price_per_1M = pricing.output_per_1m_after or pricing.output_per_1m
        else:
            # standard context pricing
            input_price_per_1M = pricing.input_per_1m
            output_price_per_1M = pricing.output_per_1m

        input_cost = self.input_tokens * input_price_per_1M
        output_cost = self.output_tokens * output_price_per_1M

        return (input_cost + output_cost) / 1_000_000
