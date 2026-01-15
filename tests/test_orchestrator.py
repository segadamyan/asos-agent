"""Tests for the Orchestrator."""

import pytest

from orchestration import Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test basic orchestrator initialization."""
    orchestrator = Orchestrator(name="TestOrchestrator")
    assert orchestrator.name == "TestOrchestrator"
    assert orchestrator.list_agents() == []


@pytest.mark.asyncio
async def test_orchestrator_agent_registration():
    """Test registering agents with orchestrator."""
    from orchestration import MathAgent

    orchestrator = Orchestrator()
    math_agent = MathAgent()

    orchestrator.register_agent("math", math_agent)
    assert "math" in orchestrator.list_agents()
    assert len(orchestrator.list_agents()) == 1
