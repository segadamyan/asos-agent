"""Tests for delegation system and Math Agent."""

import pytest

from orchestration import MathAgent, Orchestrator


@pytest.mark.asyncio
async def test_math_agent_direct(check_openai_api_key):
    """Test the math agent directly."""
    math_agent = MathAgent()

    problems = [
        "Calculate the derivative of f(x) = 3x^2 + 2x - 5",
        "Solve the quadratic equation: 2x^2 - 7x + 3 = 0",
        "What is the integral of cos(x)dx?",
    ]

    for problem in problems:
        result = await math_agent.solve(problem)
        assert result is not None
        assert result.content is not None
        assert len(result.content) > 0


@pytest.mark.asyncio
async def test_delegation_system(check_openai_api_key):
    """Test the delegation system with orchestrator."""
    # Create orchestrator (has built-in delegate tool)
    orchestrator = Orchestrator(name="MainOrchestrator")

    # Register math agent with orchestrator
    math_agent = MathAgent()
    orchestrator.register_agent("math", math_agent)

    assert len(orchestrator.list_agents()) > 0

    # Test queries that should trigger delegation
    queries = [
        "I need to solve this calculus problem: find the derivative of 5x^3 - 2x + 7",
        "Can you help me with this equation: x^2 + 5x + 6 = 0",
        "Calculate the sum of the first 10 fibonacci numbers",
    ]

    for query in queries:
        result = await orchestrator.execute(query)
        assert result is not None
        assert result.content is not None


@pytest.mark.asyncio
async def test_mixed_workflow(check_openai_api_key):
    """Test orchestrator handling both delegated and direct tasks."""
    # Create orchestrator (has built-in delegate tool)
    orchestrator = Orchestrator(name="MainOrchestrator")

    # Register math agent
    math_agent = MathAgent()
    orchestrator.register_agent("math", math_agent)

    # Mixed queries
    queries = [
        "What is machine learning?",  # Should answer directly
        "Calculate the area of a circle with radius 7",  # Should delegate to math
        "Explain the concept of recursion",  # Should answer directly
        "Solve: 3x + 7 = 22",  # Should delegate to math
    ]

    for query in queries:
        result = await orchestrator.execute(query)
        assert result is not None
        assert result.content is not None
