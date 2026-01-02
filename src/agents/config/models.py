from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from agents.providers.models.base import ProvidersEnum
from agents.utils.logs.config import logger


class ModelCapability(str, Enum):
    """Enumeration of model capabilities"""

    THINKING = "thinking"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"


class TokenConfig(BaseModel):
    """Token configuration for a model"""

    encoding: str = Field(description="Tiktoken encoding to use")
    base_tokens: int = Field(description="Base token overhead for requests")
    max_tokens: int = Field(description="Maximum context window size")
    output_tokens: int = Field(description="Maximum output tokens")


class PricingConfig(BaseModel):
    """Pricing configuration for a model"""

    input_per_1m: float = Field(description="Cost per 1M input tokens in USD")
    output_per_1m: float = Field(description="Cost per 1M output tokens in USD")
    cached_input_per_1m: Optional[float] = Field(None, description="Cost per 1M cached input tokens in USD")
    cache_creation_per_1m: Optional[float] = Field(None, description="Cost per 1M cache creation tokens in USD")
    cache_read_per_1m: Optional[float] = Field(None, description="Cost per 1M cache read tokens in USD")

    threshold: Optional[int] = Field(None, description="Token threshold for tiered pricing")
    input_per_1m_after: Optional[float] = Field(None, description="Input cost after threshold")
    output_per_1m_after: Optional[float] = Field(None, description="Output cost after threshold")


class BaseModelConfiguration(BaseModel):
    """Base configuration for all models"""

    name: str = Field(description="Model name/identifier")
    display_name: str = Field(description="Human-readable model name")
    release_date: Optional[datetime] = Field(None, description="Model release date")
    deprecation_date: Optional[datetime] = Field(None, description="Model deprecation date")

    token_config: TokenConfig = Field(description="Token and context configuration")
    pricing: PricingConfig = Field(description="Pricing information")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="Model capabilities")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for the model")


class OpenAIModelConfiguration(BaseModelConfiguration):
    """OpenAI-specific model configuration"""

    provider: str = Field(default=ProvidersEnum.openai, description="Provider name")


class AnthropicModelConfiguration(BaseModelConfiguration):
    """Anthropic-specific model configuration"""

    provider: str = Field(default=ProvidersEnum.anthropic, description="Provider name")


class GeminiModelConfiguration(BaseModelConfiguration):
    """Gemini-specific model configuration"""

    provider: str = Field(default=ProvidersEnum.gemini, description="Provider name")


ModelConfiguration = Union[OpenAIModelConfiguration, AnthropicModelConfiguration, GeminiModelConfiguration]


class ProviderConfiguration(BaseModel):
    """Configuration for a provider with default settings"""

    name: str = Field(description="Provider name")
    display_name: str = Field(description="Human-readable provider name")
    default_model: str = Field(description="Default model for this provider")
    default_token_config: TokenConfig = Field(description="Default token configuration")


class ModelConfigurationFactory:
    """Factory for creating model configurations"""

    _model_classes = {
        "openai": OpenAIModelConfiguration,
        "anthropic": AnthropicModelConfiguration,
        "gemini": GeminiModelConfiguration,
    }

    @classmethod
    def create(cls, provider: str, model_data: Dict) -> ModelConfiguration:
        """Create a model configuration from data."""
        model_class = cls._model_classes.get(provider)
        if not model_class:
            raise ValueError(f"Unknown provider: {provider}")

        capabilities = [ModelCapability(cap) for cap in model_data.get("capabilities", [])]
        token_config = TokenConfig(**model_data["token_config"])
        pricing = PricingConfig(**model_data["pricing"])

        config_data = {
            "name": model_data["name"],
            "display_name": model_data["display_name"],
            "token_config": token_config,
            "pricing": pricing,
            "capabilities": capabilities,
            "aliases": model_data.get("aliases", []),
            "provider": provider,
        }

        if "release_date" in model_data:
            config_data["release_date"] = model_data["release_date"]
        if "deprecation_date" in model_data:
            config_data["deprecation_date"] = model_data["deprecation_date"]

        return model_class(**config_data)

    @classmethod
    def register_provider(cls, provider_name: str, model_class):
        """Register a new provider class."""
        cls._model_classes[provider_name] = model_class


class ModelRegistry:
    """Registry for managing model configurations"""

    def __init__(self):
        self.models: Dict[str, ModelConfiguration] = {}
        self.providers: Dict[str, ProviderConfiguration] = {}
        self._yaml_config_cache: Optional[Dict] = None

    def _load_yaml_config(self) -> Optional[Dict]:
        """Load configuration from YAML file if it exists."""
        if self._yaml_config_cache is not None:
            return self._yaml_config_cache

        yaml_path = Path(__file__).parent / "models.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as file:
                    self._yaml_config_cache = yaml.safe_load(file)
                    return self._yaml_config_cache
            except Exception as e:
                logger.error(f"Failed to load YAML configuration: {e}")
        return None

    def initialize(self):
        """Initialize registry from YAML configuration."""
        yaml_config = self._load_yaml_config()
        if not yaml_config:
            return

        for provider_name, provider_models in yaml_config.get("models", {}).items():
            for model_name, model_data in provider_models.items():
                model_config = ModelConfigurationFactory.create(provider_name, model_data)
                self.models[f"{provider_name}_{model_name}"] = model_config

        for provider_name, provider_data in yaml_config.get("providers", {}).items():
            token_config = TokenConfig(**provider_data["default_token_config"])
            self.providers[provider_name] = ProviderConfiguration(
                name=provider_data["name"],
                display_name=provider_data["display_name"],
                default_model=provider_data["default_model"],
                default_token_config=token_config,
            )

    def add_model(self, model: ModelConfiguration) -> ModelConfiguration:
        """
        Add a new model configuration at runtime.

        Example:
            model_config = OpenAIModelConfiguration(
                name="gpt-3.0-pro",
                display_name="GPT 3.0 Pro",
                token_config=TokenConfig(
                    encoding="cl100k_base",
                    base_tokens=8,
                    max_tokens=200000,
                    output_tokens=8192
                ),
                pricing=PricingConfig(
                    input_per_1m=2.50,
                    output_per_1m=10.00
                ),
                capabilities=[ModelCapability.VISION, ModelCapability.STREAMING]
            )
            registry.add_model(model_config)
        """
        model_key = f"{model.provider}_{model.name}"
        self.models[model_key] = model

        logger.info(f"Added new model: {model_key}")
        return model

    def get_model(self, provider: str, model_name: str) -> Optional[ModelConfiguration]:
        """Get configuration for a specific model."""
        return self.models.get(f"{provider}_{model_name}")

    def get_provider(self, provider_name: str) -> Optional[ProviderConfiguration]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider_name)

    def find_model_by_alias(self, alias: str) -> Optional[ModelConfiguration]:
        """Find a model by its alias."""
        for config in self.models.values():
            if alias in config.aliases or alias == config.name:
                return config
        return None

    def list_models(self, provider: Optional[str] = None) -> List[ModelConfiguration]:
        """List all models, optionally filtered by provider."""
        if provider:
            return [m for m in self.models.values() if m.provider == provider]
        return list(self.models.values())


model_registry = ModelRegistry()
model_registry.initialize()

ALL_MODELS = model_registry.models
PROVIDERS = model_registry.providers
