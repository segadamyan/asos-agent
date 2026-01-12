"""Tests for provider integration."""

import pytest

from agents.core.agent import Agent
from agents.providers.models.base import History, IntelligenceProviderConfig


@pytest.mark.asyncio
async def test_openai_provider(check_openai_api_key):
    """Test OpenAI provider with NVIDIA endpoint."""
    # Create agent with NVIDIA/OpenAI compatible model
    ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

    agent = Agent(
        name="TestAgent",
        system_prompt="You are a helpful assistant.",
        history=History(),
        ip_config=ip_config,
    )

    # Test a simple query
    response = await agent.ask("What is 2+2? Answer in one sentence.")

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0
