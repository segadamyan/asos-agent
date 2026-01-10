#!/usr/bin/env python3
"""
Test script for the orchestration agents.

Run with:
    cd asos-agent
    poetry run python test_agents.py

Or with Python directly (if dependencies installed):
    python test_agents.py

Required environment variables:
    OPENAI_API_KEY - For OpenAI/NVIDIA endpoint
    ANTHROPIC_API_KEY - For Anthropic (optional)
    GOOGLE_API_KEY - For Gemini (optional)
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def check_api_keys():
    """Check if required API keys are set."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }
    
    print("=" * 60)
    print("API Key Status:")
    print("=" * 60)
    for key_name, key_value in keys.items():
        status = "✅ Set" if key_value else "❌ Not set"
        print(f"  {key_name}: {status}")
    print("=" * 60)
    
    if not keys["OPENAI_API_KEY"]:
        print("\n⚠️  Warning: OPENAI_API_KEY is not set.")
        print("   The default provider requires this key.")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return False
    return True


async def test_simple_agent():
    """Test a basic SimpleAgent."""
    print("\n" + "=" * 60)
    print("TEST 1: SimpleAgent (Basic)")
    print("=" * 60)
    
    from agents.core.simple import SimpleAgent
    from agents.providers.models.base import (
        GenerationBehaviorSettings,
        History,
        IntelligenceProviderConfig,
    )
    
    # Create a simple agent
    agent = SimpleAgent(
        name="TestAgent",
        system_prompt="You are a helpful assistant. Be concise.",
        history=History(),
        ip_config=IntelligenceProviderConfig(
            provider_name="openai",
            version="qwen/qwen3-next-80b-a3b-instruct"
        ),
    )
    
    # Test with a simple question
    query = "What is 2 + 2? Just give the number."
    print(f"\n📝 Query: {query}")
    
    try:
        gbs = GenerationBehaviorSettings(temperature=0.3, max_output_tokens=100)
        response = await agent.answer_to(query, gbs)
        print(f"✅ Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_math_agent():
    """Test the MathAgent."""
    print("\n" + "=" * 60)
    print("TEST 2: MathAgent")
    print("=" * 60)
    
    from orchestration.math_agent import MathAgent
    
    agent = MathAgent()
    
    query = "What is the derivative of x^2? Give a brief answer."
    print(f"\n📝 Query: {query}")
    
    try:
        response = await agent.solve(query)
        print(f"✅ Response: {response.content[:500]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_science_agent():
    """Test the ScienceAgent."""
    print("\n" + "=" * 60)
    print("TEST 3: ScienceAgent")
    print("=" * 60)
    
    from orchestration.science_agent import ScienceAgent
    
    agent = ScienceAgent()
    
    query = "What is H2O? One sentence answer."
    print(f"\n📝 Query: {query}")
    
    try:
        response = await agent.solve(query)
        print(f"✅ Response: {response.content[:500]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_code_agent():
    """Test the CodeAgent."""
    print("\n" + "=" * 60)
    print("TEST 4: CodeAgent")
    print("=" * 60)
    
    from orchestration.code_agent import CodeAgent
    
    agent = CodeAgent()
    
    query = "Write a Python function to add two numbers. Keep it simple."
    print(f"\n📝 Query: {query}")
    
    try:
        response = await agent.solve(query)
        print(f"✅ Response:\n{response.content[:500]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_business_law_agent():
    """Test the BusinessLawAgent with its tools."""
    print("\n" + "=" * 60)
    print("TEST 5: BusinessLawAgent (with Financial Tools)")
    print("=" * 60)
    
    from orchestration.business_law_agent import BusinessLawAgent
    
    agent = BusinessLawAgent()
    
    # Test a question that should use the financial_calculator tool
    query = """Calculate the Net Present Value (NPV) for an investment with:
    - Initial investment: $10,000 (negative cash flow)
    - Year 1: $3,000
    - Year 2: $4,000
    - Year 3: $5,000
    - Discount rate: 10%
    
    Should we invest? Keep the answer brief."""
    
    print(f"\n📝 Query: {query[:100]}...")
    
    try:
        response = await agent.solve(query)
        print(f"✅ Response:\n{response.content[:600]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_orchestrator():
    """Test the Orchestrator with registered agents."""
    print("\n" + "=" * 60)
    print("TEST 6: Orchestrator (Multi-Agent Routing)")
    print("=" * 60)
    
    from orchestration.orchestrator import Orchestrator
    from orchestration.math_agent import MathAgent
    from orchestration.science_agent import ScienceAgent
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Register agents
    orchestrator.register_agent("math", MathAgent())
    orchestrator.register_agent("science", ScienceAgent())
    
    print(f"\n📋 Registered agents: {orchestrator.list_agents()}")
    
    # Test routing to math agent
    query = "What is the integral of x^3? Keep it brief."
    print(f"\n📝 Query: {query}")
    
    try:
        response = await orchestrator.execute(query)
        print(f"✅ Response:\n{response.content[:500]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_financial_calculator_directly():
    """Test the financial calculator tool directly."""
    print("\n" + "=" * 60)
    print("TEST 7: Financial Calculator (Direct Tool Test)")
    print("=" * 60)
    
    from orchestration.business_law_agent import BusinessLawAgent
    
    agent = BusinessLawAgent()
    
    # Test NPV calculation directly
    print("\n📝 Testing NPV calculation...")
    result = await agent.financial_calculator(
        operation="npv",
        cash_flows=[-10000, 3000, 4000, 5000],
        rate=0.10
    )
    print(f"✅ NPV Result: {result}")
    
    # Test IRR calculation directly
    print("\n📝 Testing IRR calculation...")
    result = await agent.financial_calculator(
        operation="irr",
        cash_flows=[-10000, 3000, 4000, 5000]
    )
    print(f"✅ IRR Result: {result}")
    
    # Test current ratio
    print("\n📝 Testing Current Ratio...")
    result = await agent.financial_calculator(
        operation="current_ratio",
        current_assets=500000,
        current_liabilities=300000
    )
    print(f"✅ Current Ratio: {result}")
    
    return True


async def test_statistical_calculator_directly():
    """Test the statistical calculator tool directly."""
    print("\n" + "=" * 60)
    print("TEST 8: Statistical Calculator (Direct Tool Test)")
    print("=" * 60)
    
    from orchestration.business_law_agent import BusinessLawAgent
    
    agent = BusinessLawAgent()
    
    # Test mean
    print("\n📝 Testing Mean calculation...")
    result = await agent.statistical_calculator(
        operation="mean",
        data=[10, 20, 30, 40, 50]
    )
    print(f"✅ Mean Result: {result}")
    
    # Test regression
    print("\n📝 Testing Linear Regression...")
    result = await agent.statistical_calculator(
        operation="regression",
        x_data=[1, 2, 3, 4, 5],
        y_data=[2, 4, 5, 4, 5]
    )
    print(f"✅ Regression Result: {result}")
    
    # Test correlation
    print("\n📝 Testing Correlation...")
    result = await agent.statistical_calculator(
        operation="correlation",
        x_data=[1, 2, 3, 4, 5],
        y_data=[2, 4, 5, 4, 5]
    )
    print(f"✅ Correlation Result: {result}")
    
    return True


async def run_quick_test():
    """Run only the direct tool tests (no API calls)."""
    print("\n" + "🚀" * 20)
    print("QUICK TEST: Tool Functions Only (No API Calls)")
    print("🚀" * 20)
    
    results = []
    results.append(("Financial Calculator", await test_financial_calculator_directly()))
    results.append(("Statistical Calculator", await test_statistical_calculator_directly()))
    
    return results


async def run_all_tests():
    """Run all tests."""
    print("\n" + "🚀" * 20)
    print("RUNNING ALL TESTS")
    print("🚀" * 20)
    
    results = []
    
    # Quick tests first (no API)
    results.append(("Financial Calculator", await test_financial_calculator_directly()))
    results.append(("Statistical Calculator", await test_statistical_calculator_directly()))
    
    # API tests
    results.append(("SimpleAgent", await test_simple_agent()))
    results.append(("MathAgent", await test_math_agent()))
    results.append(("ScienceAgent", await test_science_agent()))
    results.append(("CodeAgent", await test_code_agent()))
    results.append(("BusinessLawAgent", await test_business_law_agent()))
    results.append(("Orchestrator", await test_orchestrator()))
    
    return results


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)


async def main():
    """Main entry point."""
    print("\n" + "🧪" * 20)
    print("ASOS-AGENT TEST SUITE")
    print("🧪" * 20)
    
    # Check arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Run only tool tests (no API calls)
            results = await run_quick_test()
            print_summary(results)
            return
        elif sys.argv[1] == "--help":
            print("""
Usage:
    python test_agents.py          # Run all tests (requires API keys)
    python test_agents.py --quick  # Run only tool tests (no API calls)
    python test_agents.py --help   # Show this help
            """)
            return
    
    # Check API keys
    if not check_api_keys():
        print("\n💡 Tip: Run with --quick to test tools without API calls:")
        print("   python test_agents.py --quick")
        
        response = input("\nContinue with API tests anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nRunning quick tests only...")
            results = await run_quick_test()
            print_summary(results)
            return
    
    # Run all tests
    results = await run_all_tests()
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())

