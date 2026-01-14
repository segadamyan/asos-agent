"""Tests for the Orchestrator."""

from datetime import datetime

import pytest

from agents.tools.base import ToolDefinition
from orchestration import Orchestrator


# Define some example tools
async def calculate(operation: str, a: float, b: float) -> float:
    """Performs basic arithmetic calculations"""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Cannot divide by zero",
    }
    return operations.get(operation, "Invalid operation")


async def search_web(query: str) -> str:
    """Simulates a web search"""
    return f"Search results for '{query}': [Mock result - in production this would perform actual search]"


async def get_current_time() -> str:
    """Gets the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def orchestrator_with_tools():
    """Create orchestrator with test tools."""
    calculator_tool = ToolDefinition(
        name="calculate",
        description="Performs basic arithmetic calculations (add, subtract, multiply, divide)",
        args_description={
            "operation": "The operation to perform: add, subtract, multiply, or divide",
            "a": "First number",
            "b": "Second number",
        },
        args_schema={
            "operation": {"type": "string"},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        tool=calculate,
    )

    search_tool = ToolDefinition(
        name="search_web",
        description="Searches the web for information",
        args_description={"query": "The search query"},
        args_schema={"query": {"type": "string"}},
        tool=search_web,
    )

    time_tool = ToolDefinition(
        name="get_current_time",
        description="Gets the current date and time",
        args_description={},
        args_schema={},
        tool=get_current_time,
    )

    return Orchestrator(
        name="MainOrchestrator",
        tools=[calculator_tool, search_tool, time_tool],
        additional_prompt="Be concise and efficient in your responses.",
    )


@pytest.mark.asyncio
async def test_simple_calculation(check_openai_api_key, orchestrator_with_tools):
    """Test simple calculation."""
    response = await orchestrator_with_tools.execute("What is 156 multiplied by 23?")
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_multistep_calculation(check_openai_api_key, orchestrator_with_tools):
    """Test multi-step calculation."""
    orchestrator_with_tools.clear_history()
    response = await orchestrator_with_tools.execute("Calculate (45 + 55) and then divide by 10")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_get_time(check_openai_api_key, orchestrator_with_tools):
    """Test using time tool."""
    orchestrator_with_tools.clear_history()
    response = await orchestrator_with_tools.execute("What time is it right now?")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_web_search(check_openai_api_key, orchestrator_with_tools):
    """Test web search simulation."""
    orchestrator_with_tools.clear_history()
    response = await orchestrator_with_tools.execute("Search for information about Python asyncio")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_complex_workflow(check_openai_api_key, orchestrator_with_tools):
    """Test complex workflow with multiple tools."""
    orchestrator_with_tools.clear_history()
    response = await orchestrator_with_tools.execute("Calculate 25 * 4, then search for the result")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_no_tools_needed(check_openai_api_key, orchestrator_with_tools):
    """Test orchestrator without using tools."""
    orchestrator_with_tools.clear_history()
    response = await orchestrator_with_tools.execute("Explain what an orchestrator is in AI systems")
    assert response is not None
    assert response.content is not None
