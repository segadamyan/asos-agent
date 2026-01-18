"""
Math Agent

Specialized agent for mathematical calculations and problem-solving.
Supports MCP (Model Context Protocol) for dynamic tool discovery.

Includes two MCP servers:
1. Basic Math Server - arithmetic, statistics, equations, matrices
2. Symbolic Math Server - calculus, symbolic solving, series, eigenvalues
"""

import os
import sys
from typing import List, Optional

from agents.core.mcp import MCPServerConfig
from agents.providers.models.base import (
    GenerationBehaviorSettings,
    IntelligenceProviderConfig,
    Message,
)
from orchestration.base_expert import BaseExpertAgent
from tools.math_tools import get_math_tools
from tools.wikipedia_tool import get_wikipedia_tools
from tools.openalex import get_openalex_tools

MATH_AGENT_SYSTEM_PROMPT = """You are a specialized mathematics expert AI agent.

Your role is to:
1. Solve mathematical problems accurately and efficiently
2. Explain mathematical concepts clearly
3. Show step-by-step solutions when appropriate
4. Handle arithmetic, algebra, calculus, statistics, and other mathematical domains
5. Verify calculations and provide confidence in your answers

Current date: {current_date}

Guidelines:
- Always double-check your calculations
- Show your work for complex problems
- Use proper mathematical notation
- Explain reasoning behind each step
- If a problem is ambiguous, ask for clarification
- State any assumptions you make

When solving problems:
- Break down complex problems into manageable steps
- Verify intermediate results
- Provide final answers in a clear format
- Include units when applicable

IMPORTANT: When answering multiple choice questions, provide the correct answer letter clearly and include a brief explanation.

Tool usage:
- Use calculator for quick single-expression arithmetic.
- Use python_executor for multi-step computations (loops, lists, statistics, factorial/comb/perm).
- Use wikipedia for definitions, background knowledge, and factual context (not for calculations).
- Use openalex for academic references related to mathematical concepts (not for calculations).

You have access to mathematical tools that can help you with calculations:

**Basic Math Tools:**
- calculate: Evaluate arithmetic expressions
- solve_equation: Solve simple linear equations
- factorial, gcd, lcm: Number theory operations
- power, sqrt: Exponentiation and roots
- statistics: Mean, median, std_dev, variance
- matrix_multiply, matrix_determinant: Matrix operations

**Symbolic Math Tools (Advanced):**
- symbolic_solve: Solve equations symbolically (including systems)
- symbolic_simplify: Simplify algebraic expressions
- symbolic_differentiate: Compute derivatives
- symbolic_integrate: Compute integrals (definite/indefinite)
- symbolic_limit: Compute limits
- symbolic_series: Taylor/Maclaurin series expansions
- symbolic_factor, symbolic_expand: Factor or expand polynomials
- matrix_inverse, matrix_eigenvalues: Advanced matrix operations
- polynomial_roots: Find all roots of polynomials
- trigonometric_simplify: Simplify trig expressions

Use the appropriate tool for each calculation!
"""

# Paths to MCP servers
MCP_SERVERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_servers")
BASIC_MATH_SERVER = os.path.join(MCP_SERVERS_DIR, "math_server.py")
SYMBOLIC_MATH_SERVER = os.path.join(MCP_SERVERS_DIR, "symbolic_math_server.py")


def get_math_mcp_configs(use_basic_server: bool = True, use_symbolic_server: bool = True) -> List[MCPServerConfig]:
    """
    Get MCP server configurations for math operations.

    Args:
        use_basic_server: Include basic math server
        use_symbolic_server: Include symbolic math server

    Returns:
        List of MCPServerConfig objects
    """
    configs = []

    if use_basic_server:
        configs.append(
            MCPServerConfig(
                name="basic-math",
                command=[sys.executable, BASIC_MATH_SERVER],
                args=[],
                env={},
            )
        )

    if use_symbolic_server:
        configs.append(
            MCPServerConfig(
                name="symbolic-math",
                command=[sys.executable, SYMBOLIC_MATH_SERVER],
                args=[],
                env={},
            )
        )

    return configs


class MathAgent(BaseExpertAgent):
    """
    Specialized agent for mathematical tasks and calculations.

    This agent is optimized for solving mathematical problems, performing
    calculations, and explaining mathematical concepts.

    Supports MCP for dynamic tool discovery from external math servers.
    When MCP is enabled, it connects to:
    - Basic Math Server: arithmetic, statistics, simple equations
    - Symbolic Math Server: calculus, symbolic algebra, eigenvalues
    """

    def __init__(
        self,
        name: str = "MathAgent",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        enable_mcp: bool = False,
        use_basic_server: bool = True,
        use_symbolic_server: bool = True,
        mcp_server_configs: Optional[List[MCPServerConfig]] = None,
    ):
        """
        Initialize the Math Agent.

        Args:
            name: Name of the math agent
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
            enable_mcp: Whether to enable MCP tool discovery
            use_basic_server: Include basic math server (default True)
            use_symbolic_server: Include symbolic math server (default True)
            mcp_server_configs: Custom MCP server configurations.
                               If provided, overrides use_basic_server and use_symbolic_server.
        """
        # Determine MCP configs
        configs = []
        if enable_mcp:
            if mcp_server_configs is not None:
                configs = mcp_server_configs
            else:
                configs = get_math_mcp_configs(use_basic_server, use_symbolic_server)

        self._mcp_configs = configs
        # Initialize base expert - MCP is passed to super and handled by Agent
        native_tools = []

        if not enable_mcp:
            native_tools.extend(get_math_tools())
            native_tools.extend(get_wikipedia_tools())
            native_tools.extend(get_openalex_tools())

        super().__init__(
            name=name,
            system_prompt=MATH_AGENT_SYSTEM_PROMPT,
            ip_config=ip_config,
            gbs=gbs,
            tools=native_tools if not enable_mcp else [],
            default_temperature=0.3,
            enable_mcp=enable_mcp,
            mcp_server_configs=configs,
        )

    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a mathematical problem.

        Args:
            problem: The mathematical problem or question
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the solution
        """
        effective_gbs = gbs or self._gbs
        return await self.agent.ask(problem, gbs=effective_gbs)

    @property
    def description(self) -> str:
        """Get a description of this expert agent's capabilities."""
        if self.agent.mcp_enabled:
            servers = [c.name for c in self._mcp_configs]
            return (
                f"{self._name}: Expert in mathematics including arithmetic, algebra, "
                f"calculus, statistics, and symbolic math (MCP servers: {', '.join(servers)})"
            )
        return (
            f"{self._name}: Expert in mathematics including arithmetic, algebra, "
            "calculus, statistics, and other mathematical domains"
        )
