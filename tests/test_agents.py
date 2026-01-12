"""
Tests for orchestration agents.

Run with:
    poetry run pytest tests/test_agents.py
    poetry run pytest tests/test_agents.py -v
    poetry run pytest tests/test_agents.py::test_simple_agent -v
"""

import pytest

from agents.core.agent import Agent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig
from orchestration.business_law_agent import BusinessLawAgent
from orchestration.code_agent import CodeAgent
from orchestration.math_agent import MathAgent
from orchestration.orchestrator import Orchestrator
from orchestration.science_agent import ScienceAgent


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_simple_agent(check_openai_api_key):
    """Test a basic Agent."""
    # Create an agent
    agent = Agent(
        name="TestAgent",
        system_prompt="You are a helpful assistant. Be concise.",
        history=History(),
        ip_config=IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct"),
    )

    # Test with a simple question
    query = "What is 2 + 2? Just give the number."
    gbs = GenerationBehaviorSettings(temperature=0.3, max_output_tokens=100)
    response = await agent.ask(query, gbs)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_math_agent(check_openai_api_key):
    """Test the MathAgent."""
    agent = MathAgent()
    query = "What is the derivative of x^2? Give a brief answer."
    response = await agent.solve(query)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_science_agent(check_openai_api_key):
    """Test the ScienceAgent."""
    agent = ScienceAgent()
    query = "What is H2O? One sentence answer."
    response = await agent.solve(query)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_code_agent(check_openai_api_key):
    """Test the CodeAgent."""
    agent = CodeAgent()
    query = "Write a Python function to add two numbers. Keep it simple."
    response = await agent.solve(query)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_business_law_agent(check_openai_api_key):
    """Test the BusinessLawAgent with its tools."""
    agent = BusinessLawAgent()

    query = """Calculate the Net Present Value (NPV) for an investment with:
    - Initial investment: $10,000 (negative cash flow)
    - Year 1: $3,000
    - Year 2: $4,000
    - Year 3: $5,000
    - Discount rate: 10%
    
    Should we invest? Keep the answer brief."""

    response = await agent.solve(query)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_orchestrator(check_openai_api_key):
    """Test the Orchestrator with registered agents."""
    # Create orchestrator
    orchestrator = Orchestrator()

    # Register agents
    orchestrator.register_agent("math", MathAgent())
    orchestrator.register_agent("science", ScienceAgent())

    # Test routing to math agent
    query = "What is the integral of x^3? Keep it brief."
    response = await orchestrator.execute(query)

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_financial_calculator_directly():
    """Test the financial calculator tool directly."""
    agent = BusinessLawAgent()

    # Test NPV calculation
    result = await agent.financial_calculator(operation="npv", cash_flows=[-10000, 3000, 4000, 5000], rate=0.10)
    assert result is not None
    result_str = str(result)
    assert "NPV" in result_str or "npv" in result_str.lower()

    # Test IRR calculation
    result = await agent.financial_calculator(operation="irr", cash_flows=[-10000, 3000, 4000, 5000])
    assert result is not None
    result_str = str(result)
    assert "IRR" in result_str or "irr" in result_str.lower()

    # Test current ratio
    result = await agent.financial_calculator(
        operation="current_ratio", current_assets=500000, current_liabilities=300000
    )
    assert result is not None
    result_str = str(result)
    assert "ratio" in result_str.lower()


@pytest.mark.asyncio
async def test_statistical_calculator_directly():
    """Test the statistical calculator tool directly."""
    agent = BusinessLawAgent()

    # Test mean
    result = await agent.statistical_calculator(operation="mean", data=[10, 20, 30, 40, 50])
    assert result is not None
    result_str = str(result)
    assert "30" in result_str  # Mean should be 30

    # Test regression
    result = await agent.statistical_calculator(operation="regression", x_data=[1, 2, 3, 4, 5], y_data=[2, 4, 5, 4, 5])
    assert result is not None
    result_str = str(result)
    assert "slope" in result_str.lower() or "intercept" in result_str.lower()

    # Test correlation
    result = await agent.statistical_calculator(operation="correlation", x_data=[1, 2, 3, 4, 5], y_data=[2, 4, 5, 4, 5])
    assert result is not None
    result_str = str(result)
    assert "result" in result_str.lower()  # Check for result key in the dict
