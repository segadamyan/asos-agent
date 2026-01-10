#!/usr/bin/env python3
"""
Test MCP Math Agent

This script tests the MathAgent with MCP support, connecting to both
the basic math server and the symbolic math server.

Usage:
    poetry run python test_mcp_math.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_basic_math_server():
    """Test the basic MCP math server tools."""
    print("\n" + "=" * 60)
    print("Testing Basic Math MCP Server")
    print("=" * 60)
    
    from agents.core.mcp import MCPDiscovery, MCPServerConfig
    
    math_server_path = os.path.join(
        os.path.dirname(__file__),
        "src", "mcp_servers", "math_server.py"
    )
    
    config = MCPServerConfig(
        name="basic-math",
        command=[sys.executable, math_server_path],
        args=[],
    )
    
    async with MCPDiscovery() as discovery:
        # Connect and discover tools
        print("\n1. Connecting to basic math server...")
        success = await discovery.connect_server(config)
        
        if not success:
            print("❌ Failed to connect")
            return False
        
        print("✅ Connected")
        
        # Discover tools
        tools = await discovery.discover_tools()
        print(f"✅ Discovered {len(tools)} tools")
        
        # Get client
        client = discovery.get_client("basic-math")
        
        # Test calculate
        print("\n2. Testing 'calculate'...")
        result = await client.call_tool("calculate", {"expression": "sqrt(144) + 2^3"})
        print(f"   √144 + 2³ = {result.content}")
        
        # Test statistics
        print("\n3. Testing 'statistics'...")
        result = await client.call_tool("statistics", {"data": [10, 20, 30, 40, 50]})
        print(f"   Stats: {result.content[:100]}...")
        
        # Test solve_equation
        print("\n4. Testing 'solve_equation'...")
        result = await client.call_tool("solve_equation", {"equation": "3x - 9 = 0"})
        print(f"   3x - 9 = 0 → {result.content}")
        
    print("\n✅ Basic math server tests passed!")
    return True


async def test_symbolic_math_server():
    """Test the symbolic MCP math server tools."""
    print("\n" + "=" * 60)
    print("Testing Symbolic Math MCP Server")
    print("=" * 60)
    
    from agents.core.mcp import MCPDiscovery, MCPServerConfig
    
    symbolic_server_path = os.path.join(
        os.path.dirname(__file__),
        "src", "mcp_servers", "symbolic_math_server.py"
    )
    
    config = MCPServerConfig(
        name="symbolic-math",
        command=[sys.executable, symbolic_server_path],
        args=[],
    )
    
    async with MCPDiscovery() as discovery:
        # Connect and discover tools
        print("\n1. Connecting to symbolic math server...")
        success = await discovery.connect_server(config)
        
        if not success:
            print("❌ Failed to connect")
            return False
        
        print("✅ Connected")
        
        # Discover tools
        tools = await discovery.discover_tools()
        print(f"✅ Discovered {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}")
        
        # Get client
        client = discovery.get_client("symbolic-math")
        
        # Test symbolic_solve
        print("\n2. Testing 'symbolic_solve' (quadratic equation)...")
        result = await client.call_tool("symbolic_solve", {
            "equations": ["x**2 - 5*x + 6"],
            "variables": ["x"]
        })
        print(f"   x² - 5x + 6 = 0 → {result.content}")
        
        # Test symbolic_differentiate
        print("\n3. Testing 'symbolic_differentiate'...")
        result = await client.call_tool("symbolic_differentiate", {
            "expression": "x**3 + 2*x**2 - 5*x + 1",
            "variable": "x"
        })
        print(f"   d/dx(x³ + 2x² - 5x + 1) = {result.content}")
        
        # Test symbolic_integrate
        print("\n4. Testing 'symbolic_integrate' (indefinite)...")
        result = await client.call_tool("symbolic_integrate", {
            "expression": "x**2 + 3*x",
            "variable": "x"
        })
        print(f"   ∫(x² + 3x)dx = {result.content}")
        
        # Test symbolic_integrate (definite)
        print("\n5. Testing 'symbolic_integrate' (definite)...")
        result = await client.call_tool("symbolic_integrate", {
            "expression": "x**2",
            "variable": "x",
            "lower_bound": "0",
            "upper_bound": "1"
        })
        print(f"   ∫₀¹ x² dx = {result.content}")
        
        # Test symbolic_limit
        print("\n6. Testing 'symbolic_limit'...")
        result = await client.call_tool("symbolic_limit", {
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0"
        })
        print(f"   lim(x→0) sin(x)/x = {result.content}")
        
        # Test symbolic_series
        print("\n7. Testing 'symbolic_series' (Taylor expansion)...")
        result = await client.call_tool("symbolic_series", {
            "expression": "exp(x)",
            "variable": "x",
            "point": "0",
            "order": 5
        })
        print(f"   e^x Taylor series: {result.content}")
        
        # Test symbolic_factor
        print("\n8. Testing 'symbolic_factor'...")
        result = await client.call_tool("symbolic_factor", {
            "expression": "x**2 - 9"
        })
        print(f"   x² - 9 = {result.content}")
        
        # Test matrix_eigenvalues
        print("\n9. Testing 'matrix_eigenvalues'...")
        result = await client.call_tool("matrix_eigenvalues", {
            "matrix": [[4, 2], [1, 3]]
        })
        print(f"   Eigenvalues of [[4,2],[1,3]]: {result.content[:150]}...")
        
        # Test polynomial_roots
        print("\n10. Testing 'polynomial_roots'...")
        result = await client.call_tool("polynomial_roots", {
            "coefficients": [1, 0, -4]  # x² - 4
        })
        print(f"   Roots of x² - 4: {result.content}")
        
        # Test trigonometric_simplify
        print("\n11. Testing 'trigonometric_simplify'...")
        result = await client.call_tool("trigonometric_simplify", {
            "expression": "sin(x)**2 + cos(x)**2"
        })
        print(f"   sin²(x) + cos²(x) = {result.content}")
        
    print("\n✅ Symbolic math server tests passed!")
    return True


async def test_combined_math_agent():
    """Test MathAgent with both servers combined (MCP only, no LLM)."""
    print("\n" + "=" * 60)
    print("Testing Combined MCP Servers (Both Basic + Symbolic)")
    print("=" * 60)
    
    from agents.core.mcp import MCPDiscovery, MCPServerConfig
    
    basic_server_path = os.path.join(
        os.path.dirname(__file__),
        "src", "mcp_servers", "math_server.py"
    )
    symbolic_server_path = os.path.join(
        os.path.dirname(__file__),
        "src", "mcp_servers", "symbolic_math_server.py"
    )
    
    configs = [
        MCPServerConfig(name="basic-math", command=[sys.executable, basic_server_path]),
        MCPServerConfig(name="symbolic-math", command=[sys.executable, symbolic_server_path]),
    ]
    
    async with MCPDiscovery() as discovery:
        # Connect to both servers
        print("\n1. Connecting to both servers...")
        for config in configs:
            success = await discovery.connect_server(config)
            print(f"   {config.name}: {'✅' if success else '❌'}")
        
        # Discover all tools
        print("\n2. Discovering tools from all servers...")
        await discovery.discover_tools()
        tools = discovery.convert_to_tool_definitions()
        
        print(f"   Total tools: {len(tools)}")
        print(f"   Connected servers: {discovery.connected_servers}")
        
        # Verify we have tools from both servers
        basic_tools = [t for t in discovery.available_tools if t in ['calculate', 'factorial', 'statistics']]
        symbolic_tools = [t for t in discovery.available_tools if t.startswith('symbolic_') or t == 'polynomial_roots']
        
        print(f"\n   Basic math tools: {len(basic_tools)}")
        print(f"   Symbolic math tools: {len(symbolic_tools)}")
        
        if len(basic_tools) > 0 and len(symbolic_tools) > 0:
            print("\n✅ Both servers providing tools!")
        else:
            print("\n❌ Missing tools from one or more servers")
            return False
    
    print("\n✅ Combined servers test passed!")
    return True


async def test_math_agent_with_llm():
    """Test MathAgent with MCP and LLM (requires API key)."""
    print("\n" + "=" * 60)
    print("Testing MathAgent with LLM Integration")
    print("=" * 60)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping LLM test.")
        return True
    
    from orchestration.math_agent import MathAgent
    from agents.providers.models.base import GenerationBehaviorSettings
    
    async with MathAgent(name="LLMMathAgent", enable_mcp=True) as agent:
        print(f"\n✅ MathAgent with LLM initialized")
        print(f"   Tools: {len(agent.available_tools)}")
        
        gbs = GenerationBehaviorSettings(temperature=0.2, max_tokens=500)
        
        # Test with derivative
        print("\n1. Testing calculus problem...")
        result = await agent.solve(
            "What is the derivative of x³ + 2x² - 5x + 1? Use the symbolic_differentiate tool.",
            gbs=gbs
        )
        print(f"   Response: {result.content[:300]}...")
        
        # Test with integration
        print("\n2. Testing integration...")
        result = await agent.solve(
            "Compute the definite integral of x² from 0 to 1 using the symbolic_integrate tool.",
            gbs=gbs
        )
        print(f"   Response: {result.content[:300]}...")
    
    print("\n✅ LLM integration test passed!")
    return True


async def main():
    """Run all MCP tests."""
    print("\n" + "=" * 60)
    print("🧮 MCP Math Agent Test Suite (Basic + Symbolic)")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic math server
    try:
        results["Basic Math Server"] = await test_basic_math_server()
    except Exception as e:
        print(f"\n❌ Basic math server test failed: {e}")
        results["Basic Math Server"] = False
    
    # Test 2: Symbolic math server
    try:
        results["Symbolic Math Server"] = await test_symbolic_math_server()
    except Exception as e:
        print(f"\n❌ Symbolic math server test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Symbolic Math Server"] = False
    
    # Test 3: Combined agent
    try:
        results["Combined MathAgent"] = await test_combined_math_agent()
    except Exception as e:
        print(f"\n❌ Combined agent test failed: {e}")
        results["Combined MathAgent"] = False
    
    # Test 4: LLM integration (optional)
    try:
        results["LLM Integration"] = await test_math_agent_with_llm()
    except Exception as e:
        print(f"\n❌ LLM integration test failed: {e}")
        results["LLM Integration"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
