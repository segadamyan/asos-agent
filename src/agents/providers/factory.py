"""
Provider Factory Module

This module implements the factory pattern for creating LLM providers.
The ProviderFactory class simplifies the instantiation of different LLM provider
implementations (OpenAI, Anthropic, Gemini) by abstracting away the specific
initialization details. It automatically configures providers with
tool definitions based on the provider configuration.
This centralized factory approach enables easy switching between different LLM
backends throughout the application.
"""

from agents.providers.anthropic import AnthropicProvider
from agents.providers.gemini import GeminiProvider
from agents.providers.models.base import BaseProvider, IntelligenceProviderConfig
from agents.providers.openai import OpenAIProvider
from agents.tools.base import ToolDefinition


class ProviderFactory:
    def create(
        self,
        system_prompt: str,
        ip_config: IntelligenceProviderConfig,
        tools: list[ToolDefinition] = None,
    ) -> BaseProvider:
        provider_name = ip_config.provider_name
        if provider_name == "openai":
            return OpenAIProvider(
                system_prompt,
                ip_config.version,
                tools=tools,
            )
        elif provider_name == "anthropic":
            return AnthropicProvider(
                system_prompt,
                ip_config.version,
                tools=tools,
            )
        elif provider_name == "gemini":
            return GeminiProvider(
                system_prompt,
                ip_config.version,
                tools=tools,
            )
        raise NotImplementedError(f"Provider {provider_name} is not implemented")
