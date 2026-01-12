"""Tests for tool calling functionality."""

import pytest

from agents.core.agent import Agent
from agents.providers.models.base import History, IntelligenceProviderConfig
from agents.tools.base import ToolDefinition


# Define a simple calculator tool
async def calculate(operation: str, a: float, b: float) -> float:
    """Performs basic arithmetic calculations"""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Cannot divide by zero",
    }
    return operations.get(operation, "Invalid operation")


# Define a weather tool
async def get_weather(city: str) -> str:
    """Gets the current weather for a city"""
    return f"The weather in {city} is sunny and 72°F"


@pytest.fixture
def calculator_tool():
    """Calculator tool fixture."""
    return ToolDefinition(
        name="calculate",
        description="Performs basic arithmetic calculations",
        args_description={
            "operation": "The operation to perform: add, subtract, multiply, or divide",
            "a": "First number",
            "b": "Second number",
        },
        args_schema={"operation": {"type": "string"}, "a": {"type": "number"}, "b": {"type": "number"}},
        tool=calculate,
    )


@pytest.fixture
def weather_tool():
    """Weather tool fixture."""
    return ToolDefinition(
        name="get_weather",
        description="Gets the current weather for a city",
        args_description={"city": "The name of the city to get weather for"},
        args_schema={"city": {"type": "string"}},
        tool=get_weather,
    )


@pytest.fixture
def agent_with_tools(calculator_tool, weather_tool):
    """Create agent with test tools."""
    ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

    return Agent(
        name="ToolTestAgent",
        system_prompt="You are a helpful assistant with access to tools. Use them when appropriate.",
        history=History(),
        ip_config=ip_config,
        tools=[calculator_tool, weather_tool],
    )


@pytest.mark.asyncio
async def test_calculator_tool(check_openai_api_key, agent_with_tools):
    """Test calculator tool."""
    response = await agent_with_tools.ask("What is 25 multiplied by 4?")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_weather_tool(check_openai_api_key, agent_with_tools):
    """Test weather tool."""
    agent_with_tools.history.clear()
    response = await agent_with_tools.ask("What's the weather in Paris?")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_multiple_tool_calls(check_openai_api_key, agent_with_tools):
    """Test multiple tool calls."""
    agent_with_tools.history.clear()
    response = await agent_with_tools.ask("Calculate 10 + 5 and then multiply the result by 3")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_no_tool_needed(check_openai_api_key, agent_with_tools):
    """Test when no tool is needed."""
    agent_with_tools.history.clear()
    response = await agent_with_tools.ask("Hello, how are you?")
    assert response is not None
    assert response.content is not None
    # When no tool is needed, tool_calls should be empty or zero
