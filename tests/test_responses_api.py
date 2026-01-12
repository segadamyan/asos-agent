"""Tests for OpenAI API compatibility."""

import os

import pytest
from openai import AsyncOpenAI


@pytest.mark.asyncio
async def test_chat_completions_api(check_openai_api_key):
    """Test the chat completions API."""
    api_key = os.getenv("OPENAI_API_KEY")

    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    completion = await client.chat.completions.create(
        model="qwen/qwen3-next-80b-a3b-instruct",
        messages=[{"role": "user", "content": "What is 2+2? Answer in one sentence."}],
        temperature=0.6,
        max_tokens=1024,
    )

    assert completion is not None
    assert completion.choices is not None
    assert len(completion.choices) > 0
    assert completion.choices[0].message.content is not None
