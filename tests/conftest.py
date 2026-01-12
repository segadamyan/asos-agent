"""Pytest configuration and fixtures for asos-agent tests."""

import os

import pytest


@pytest.fixture(scope="session")
def check_openai_api_key():
    """Check if OPENAI_API_KEY is set."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture(scope="session")
def check_anthropic_api_key():
    """Check if ANTHROPIC_API_KEY is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture(scope="session")
def check_google_api_key():
    """Check if GOOGLE_API_KEY is set."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OPENAI_API_KEY from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key
