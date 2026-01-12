"""
Tests for orchestrator with built-in delegation.
"""

import pytest

from orchestration import MathAgent, Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_delegation(check_openai_api_key):
    """Test orchestrator with built-in delegation."""
    # Create orchestrator
    orchestrator = Orchestrator(name="MainOrchestrator")

    # Register math agent
    math_agent = MathAgent()
    orchestrator.register_agent("math", math_agent)

    assert len(orchestrator.list_agents()) > 0

    # Test queries
    queries = [
        "What is recursion?",  # Direct response
        "Solve the equation: 2x^2 - 5x + 2 = 0",  # Should delegate to math
        "Calculate the integral of 2x dx",  # Should delegate to math
    ]

    for query in queries:
        result = await orchestrator.execute(query)
        assert result is not None
        assert result.content is not None
