# ASOS Agent

**Agentic Science Orchestration System (ASOS)** - A sophisticated multi-agent framework for AI-driven task orchestration, specialized problem-solving, and benchmark evaluation.

## 🌟 Features

- **Multi-Provider Support**: Seamlessly work with OpenAI, Anthropic, and Google Gemini models
- **Specialized Expert Agents**: Domain-specific agents for mathematics, science, and coding
- **Intelligent Orchestration**: Coordinate complex workflows with task delegation
- **Benchmark Integration**: Built-in support for MMLU-Redux and HLE (Humanity's Last Exam) benchmarks
- **Tool Execution**: Extensible tool system for enhanced agent capabilities
- **Conversation Management**: Automatic history tracking and token management
- **MCP Support**: Integration with Model Context Protocol servers

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Expert Agents](#expert-agents)
- [MCP Support](#-mcp-model-context-protocol-support)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### Setup

```bash
# Clone the repository
git clone https://github.com/segadamyan/asos-agent.git
cd asos-agent

# Install dependencies using Poetry
poetry install

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## 🎯 Quick Start

### Basic Agent Usage

```python
import asyncio
from agents.core.agent import Agent
from agents.providers.models.base import History, IntelligenceProviderConfig

async def main():
    # Create a simple agent
    agent = Agent(
        name="MyAgent",
        system_prompt="You are a helpful AI assistant.",
        history=History(),
        ip_config=IntelligenceProviderConfig(
            provider_name="openai",
            version="gpt-4o"
        )
    )
    
    # Ask a question
    response = await agent.ask("What is the capital of France?")
    print(response.content)

asyncio.run(main())
```

### Using the Orchestrator with Expert Agents

```python
import asyncio
from orchestration import Orchestrator, MathAgent, ScienceAgent, CodeAgent

async def main():
    # Create orchestrator
    orchestrator = Orchestrator(name="MyOrchestrator")
    
    # Register specialized agents
    orchestrator.register_agent("math", MathAgent())
    orchestrator.register_agent("science", ScienceAgent())
    orchestrator.register_agent("code", CodeAgent())
    
    # Execute a complex task
    response = await orchestrator.execute(
        "Calculate the trajectory of a projectile launched at 45 degrees with initial velocity of 20 m/s"
    )
    print(response.content)

asyncio.run(main())
```

## 🏗️ Architecture

### System Overview

![ASOS Architecture](images/asos_architecture.png)

The ASOS framework is organized into distinct layers:

- **Application Layer**: User interface and application logic
- **Orchestration Layer**: Task coordination and expert agent management
- **Agents Layer**: Core agent functionality with tool execution
- **Tools Layer**: 40+ specialized tools across 8 categories
- **Provider Layer**: Multi-provider LLM integration (OpenAI, Anthropic, Google)
- **MCP Layer**: Model Context Protocol for dynamic tool discovery
- **Benchmarks Layer**: Evaluation on MMLU-Redux and HLE datasets

## 🛠️ Tools System

ASOS provides **40+ specialized tools** organized into **8 categories**, enabling agents to perform complex calculations, simulations, and analysis across multiple domains.

### Tools Factory

The `ToolsFactory` provides centralized access to all tools:

```python
from tools.factory import ToolsFactory

# Get all tools
all_tools = ToolsFactory.get_all_tools()

# Get tools by category
math_tools = ToolsFactory.get_tools_by_category(["math"])
science_tools = ToolsFactory.get_tools_by_category(["physics", "chemistry"])

# Get multiple categories
tools = ToolsFactory.get_tools_by_category(["math", "physics", "chemistry"])
```

### Available Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| **Math** | 7 tools | Arithmetic, calculus, statistics, equation solving |
| **Physics** | 6 tools | Mechanics, energy, momentum, wave properties |
| **Chemistry** | 6 tools | Molar mass, pH, gas laws, stoichiometry |
| **Computer Science** | 5 tools | Binary operations, circuits, data structures |
| **Medical** | 5 tools | BMI, dosage, heart rate, fluid requirements |
| **Business/Law** | 6 tools | Financial ratios, NPV, IRR, econometrics |
| **Logic/Philosophy** | 5 tools | Logical statements, fallacies, truth tables |
| **Humanities** | 4 tools | Geography, astronomy, time zones |
| **Execution** | 1 tool | Safe Python code execution |

### Math Tools

```python
from tools.math_tools import get_math_tools

# Available tools:
# - calculator: Safe expression evaluator (2*(3+4)**2, sqrt(16), sin(pi/2))
# - calculate: Arithmetic, trigonometry, logarithms
# - statistical_calc: Mean, median, std_dev, variance, correlation, t-test
# - solve_quadratic: Solve ax² + bx + c = 0
# - calculate_derivative: Symbolic differentiation
# - calculate_integral: Definite/indefinite integration
# - python_executor: Multi-step Python code execution
```

**Example Usage:**

```python
from tools.math_tools import calculator, python_executor

# Quick calculation
result = await calculator("sqrt(144) + log10(1000)")
# Returns: {"result": 15.0}

# Multi-step computation
code = """
import math
n = 10
k = 3
result = math.comb(n, k)
print(f"C({n},{k}) = {result}")
"""
result = await python_executor(code)
# Returns: {"result": 120, "stdout": "C(10,3) = 120\n"}
```

### Physics Tools

```python
from tools.physics_tools import get_physics_tools

# Available tools:
# - calculate_force: F = ma
# - calculate_kinematics: Motion equations (v, s, a, t)
# - calculate_energy: Kinetic, potential, elastic energy
# - calculate_momentum: Linear and angular momentum
# - calculate_wave_properties: Frequency, wavelength, speed
# - calculate_thermodynamics: Heat transfer, ideal gas
```

### Chemistry Tools

```python
from tools.chemistry_tools import get_chemistry_tools

# Available tools:
# - calculate_molar_mass: Molecular weight
# - calculate_ph: pH and pOH calculations
# - calculate_concentration: Molarity, dilution
# - calculate_ideal_gas_law: PV = nRT
# - balance_equation: Chemical equation balancing
# - calculate_stoichiometry: Reaction quantities
```

### Python Executor Tool

The `python_executor` tool provides safe, sandboxed Python code execution:

**Features:**
- ✅ Pre-imported `math` and `statistics` modules
- ✅ Multi-step computations with loops and conditionals
- ✅ Print statements for debugging
- ✅ List comprehensions and data structures
- ✅ 2-second timeout protection
- ❌ No imports allowed (security)
- ❌ No file I/O or dangerous operations

**Example:**

```python
from tools.execution_tools import python_executor

code = """
# Calculate factorial and combinations
factorial_10 = math.factorial(10)
combinations = math.comb(52, 5)

print(f"10! = {factorial_10}")
print(f"C(52,5) = {combinations}")

result = {
    "factorial": factorial_10,
    "combinations": combinations
}
"""

output = await python_executor(code, timeout_seconds=2.0)
# Returns:
# {
#   "result": {"factorial": 3628800, "combinations": 2598960},
#   "stdout": "10! = 3628800\nC(52,5) = 2598960\n"
# }
```

### Calculator Tool

The `calculator` tool provides safe expression evaluation:

**Features:**
- ✅ Arithmetic: +, -, *, /, //, %, **
- ✅ Functions: sqrt, abs, round, floor, ceil, log, log10, exp
- ✅ Trigonometry: sin, cos, tan
- ✅ Constants: pi, e
- ❌ No imports or dangerous operations

**Example:**

```python
from tools.math_tools import calculator

# Simple calculation
await calculator("2 + 2")  # {"result": 4}

# Complex expression
await calculator("2*(3+4)**2 + sqrt(16)")  # {"result": 102.0}

# With functions and constants
await calculator("sin(pi/2) + cos(0)")  # {"result": 2.0}
```

### Using Tools with Agents

```python
from agents.core.agent import Agent
from tools.factory import ToolsFactory

# Create agent with specific tool categories
tools = ToolsFactory.get_tools_by_category(["math", "physics"])

agent = Agent(
    name="ScienceAgent",
    system_prompt="You are a science expert with mathematical tools.",
    history=History(),
    ip_config=IntelligenceProviderConfig(provider_name="openai", version="gpt-4o"),
    tools=tools
)

response = await agent.ask(
    "Calculate the kinetic energy of a 5kg object moving at 10 m/s"
)
```

## 🧩 Core Components

### 1. Agent

The foundational agent that handles:
- LLM provider communication
- Conversation history management
- Tool invocation
- Token tracking
- Error recovery

**Key Features:**
- Multi-provider support (OpenAI, Anthropic, Gemini)
- Automatic token limit management
- Parallel tool execution
- Provider fallback support
- Conversation forking

### 2. Orchestrator

Coordinates complex workflows by:
- Breaking down tasks into manageable steps
- Delegating to specialized expert agents
- Managing tool execution
- Maintaining workflow context

### 3. Expert Agents

Specialized agents for specific domains:

#### MathAgent
- Solves mathematical problems (algebra, calculus, statistics)
- Step-by-step explanations
- Lower temperature (0.3) for precision

#### ScienceAgent
- Physics, chemistry, biology expertise
- Evidence-based responses
- Scientific notation and units

#### CodeAgent
- Programming and computer science
- Clean, documented code
- Algorithm design and debugging
- Very low temperature (0.2) for code precision

## 🎓 Expert Agents

### Creating a Custom Expert Agent

```python
from orchestration.base_expert import BaseExpertAgent
from agents.providers.models.base import Message, GenerationBehaviorSettings
from typing import Optional

class CustomExpertAgent(BaseExpertAgent):
    def __init__(self, name: str = "CustomExpert"):
        system_prompt = """You are a specialized expert in your domain.
        Current date: {current_date}
        """
        
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            default_temperature=0.3
        )
    
    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """Solve a problem in your domain"""
        effective_gbs = gbs or self.gbs
        return await self.agent.ask(problem, gbs=effective_gbs)
    
    @property
    def description(self) -> str:
        return f"{self.name}: Expert in custom domain"
```

## 🔌 MCP (Model Context Protocol) Support

ASOS Agent supports **MCP** for dynamic tool discovery from external servers. This enables agents to use specialized computational tools without hardcoding them.

### What is MCP?

MCP (Model Context Protocol) is an open standard for AI models to communicate with external tools and services. ASOS uses **stdio transport** to spawn MCP servers as subprocesses.

### Built-in MCP Math Servers

| Server | Tools | Purpose |
|--------|-------|---------|
| **Basic Math** | 10 | Arithmetic, statistics, equations, matrices |
| **Symbolic Math** | 12 | Calculus, symbolic algebra, eigenvalues (SymPy) |

### Quick Start with MCP

```python
import asyncio
from orchestration.math_agent import MathAgent

async def main():
    # MathAgent with MCP tools enabled
    async with MathAgent(name="MathExpert", enable_mcp=True) as agent:
        # Agent now has 22 mathematical tools available
        print(f"Available tools: {agent.available_tools}")
        
        # Solve a calculus problem using symbolic tools
        result = await agent.solve(
            "Calculate the derivative of x³ + 2x² - 5x + 1"
        )
        print(result.content)

asyncio.run(main())
```

### MCP Configuration Options

```python
from orchestration.math_agent import MathAgent

# Use both math servers (default)
agent = MathAgent(enable_mcp=True)

# Use only basic math server
agent = MathAgent(enable_mcp=True, use_symbolic_server=False)

# Use only symbolic math server  
agent = MathAgent(enable_mcp=True, use_basic_server=False)

# Custom MCP server configuration
from agents.core.mcp import MCPServerConfig

custom_config = MCPServerConfig(
    name="my-math-server",
    command=["python", "my_server.py"],
    args=["--port", "8080"],
    env={"DEBUG": "1"}
)
agent = MathAgent(enable_mcp=True, mcp_server_configs=[custom_config])
```

### Available MCP Tools

#### Basic Math Server (10 tools)
| Tool | Description |
|------|-------------|
| `calculate` | Evaluate expressions (sqrt, sin, cos, log, etc.) |
| `solve_equation` | Solve linear equations |
| `factorial` | Calculate n! |
| `gcd` / `lcm` | Greatest common divisor / Least common multiple |
| `power` | Exponentiation |
| `sqrt` | Square root |
| `statistics` | Mean, median, std_dev, variance, min, max |
| `matrix_multiply` | Matrix multiplication |
| `matrix_determinant` | Matrix determinant (up to 3x3) |

#### Symbolic Math Server (12 tools)
| Tool | Description |
|------|-------------|
| `symbolic_solve` | Solve equations symbolically (including systems) |
| `symbolic_simplify` | Simplify algebraic expressions |
| `symbolic_differentiate` | Compute derivatives |
| `symbolic_integrate` | Compute integrals (definite/indefinite) |
| `symbolic_limit` | Compute limits |
| `symbolic_series` | Taylor/Maclaurin series expansion |
| `symbolic_factor` | Factor polynomials |
| `symbolic_expand` | Expand expressions |
| `matrix_inverse` | Matrix inverse |
| `matrix_eigenvalues` | Eigenvalues and eigenvectors |
| `polynomial_roots` | Find polynomial roots |
| `trigonometric_simplify` | Simplify trig expressions |

### Adding MCP to Other Agents

Any agent inheriting from `BaseExpertAgent` can use MCP:

```python
from orchestration.base_expert import BaseExpertAgent
from agents.core.mcp import MCPServerConfig

class MyCustomAgent(BaseExpertAgent):
    def __init__(self, enable_mcp=False):
        mcp_configs = [
            MCPServerConfig(
                name="my-server",
                command=["python", "my_mcp_server.py"]
            )
        ] if enable_mcp else []
        
        super().__init__(
            name="CustomAgent",
            system_prompt="Your system prompt here",
            enable_mcp=enable_mcp,
            mcp_server_configs=mcp_configs,
        )
    
    async def solve(self, problem, gbs=None):
        await self._ensure_initialized()  # Initialize MCP if enabled
        return await self.agent.answer_to(problem, gbs=gbs or self.gbs)
```

### Creating Custom MCP Servers

Create your own MCP server using the `mcp` library:

```python
# my_mcp_server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="my_tool",
            description="Description of what your tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"}
                },
                "required": ["param1"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        result = f"Processed: {arguments['param1']}"
        return [TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing MCP Integration

```bash
# Run MCP-specific tests
poetry run python test_mcp_math.py

# Expected output:
# Basic Math Server: ✅ PASSED (10 tools)
# Symbolic Math Server: ✅ PASSED (12 tools)
# Combined MathAgent: ✅ PASSED (22 tools)
```

---

## 📊 Benchmarks

### MMLU-Redux Benchmark

Evaluates agents on the MMLU-Redux dataset (3,000 questions across 30 subjects).

**Run the benchmark:**

```bash
poetry run python src/benchmarks/mmlu_redux.py
```

**Programmatic usage:**

```python
import asyncio
from src.benchmarks.mmlu_redux import run_benchmark_with_orchestrator

async def main():
    results = await run_benchmark_with_orchestrator(
        max_questions=100,
        subjects=["college_mathematics", "college_physics"],
        dataset_version="2.0"
    )
    print(f"Accuracy: {results.accuracy:.2%}")

asyncio.run(main())
```

**Configuration Options:**
- `subjects`: List of subjects to evaluate (default: all 30 subjects)
- `filter_error_types`: Filter questions by error type (e.g., `["ok"]` for correct questions only)
- `max_questions_per_subject`: Limit questions per subject
- `max_questions`: Total question limit
- `dataset_version`: "1.0" or "2.0"

### HLE (Humanity's Last Exam) Benchmark

Frontier-level benchmark with 2,500 PhD+ difficulty questions.

**Run the benchmark:**

```bash
poetry run python src/benchmarks/hle.py
```

**Programmatic usage:**

```python
import asyncio
from src.benchmarks.hle import run_benchmark_with_orchestrator

async def main():
    results = await run_benchmark_with_orchestrator(
        max_questions=50,
        categories=["Math", "Physics"],
        skip_images=True
    )
    print(f"Accuracy: {results.accuracy:.2%}")

asyncio.run(main())
```

**Configuration Options:**
- `categories`: Filter by category (`['Math', 'Physics', 'Chemistry', 'Biology/Medicine', 'Computer Science/AI', 'Engineering', 'Humanities/Social Science', 'Other']`)
- `subjects`: Filter by specific subjects
- `answer_types`: Filter by type (`['exactMatch', 'multipleChoice']`)
- `max_questions_per_category`: Limit questions per category
- `skip_images`: Skip image-based questions (default: `True`)

**Expected Performance:**
- GPT-4o/Claude 3.5: 50-60%
- Qwen3-Next-80B: 15-30%
- Random guessing: ~10%

See `src/benchmarks/HLE_ANALYSIS.md` for detailed analysis.

## ⚙️ Configuration

### Model Configuration

Models are configured in `src/agents/config/models.yaml`:

```yaml
providers:
  openai:
    default_model: "gpt-4o"
    default_token_config:
      max_tokens: 128000
      output_tokens: 16384

models:
  openai:
    gpt-4o:
      pricing:
        input_per_1m: 2.50
        output_per_1m: 10.00
      capabilities:
        - "vision"
        - "streaming"
```

### Provider Selection

```python
from agents.providers.models.base import IntelligenceProviderConfig

# OpenAI
config = IntelligenceProviderConfig(
    provider_name="openai",
    version="gpt-4o"
)

# Anthropic
config = IntelligenceProviderConfig(
    provider_name="anthropic",
    version="claude-3-5-sonnet-20241022"
)

# Google Gemini
config = IntelligenceProviderConfig(
    provider_name="gemini",
    version="gemini-2.0-flash-exp"
)
```

### Generation Behavior Settings

```python
from agents.providers.models.base import GenerationBehaviorSettings

gbs = GenerationBehaviorSettings(
    temperature=0.7,        # Creativity level (0.0-1.0)
    max_tokens=2000,        # Maximum output tokens
    top_p=0.9,             # Nucleus sampling
    frequency_penalty=0.0,  # Repetition penalty
    presence_penalty=0.0    # Topic diversity
)
```

## 💡 Examples

### Example 1: Multi-Step Problem Solving

```python
import asyncio
from orchestration import Orchestrator, MathAgent

async def main():
    orchestrator = Orchestrator()
    orchestrator.register_agent("math", MathAgent())
    
    response = await orchestrator.execute(
        "A train travels at 60 mph for 2 hours, then 80 mph for 3 hours. "
        "What is the total distance traveled and average speed?"
    )
    print(response.content)

asyncio.run(main())
```

### Example 2: Custom Tools

```python
from agents.tools.base import ToolDefinition
from agents.core.agent import Agent

async def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Weather in {city}: Sunny, 72°F"

weather_tool = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city",
    args_description={"city": "Name of the city"},
    args_schema={"city": {"type": "string"}},
    tool=get_weather
)

agent = Agent(
    name="WeatherAgent",
    system_prompt="You help users with weather information.",
    history=History(),
    ip_config=IntelligenceProviderConfig(provider_name="openai", version="gpt-4o"),
    tools=[weather_tool]
)

response = await agent.ask("What's the weather in Paris?")
```

### Example 3: Benchmark Evaluation

```python
import asyncio
from src.benchmarks.mmlu_redux import MMLUReduxBenchmark
from agents.core.agent import Agent
from agents.providers.models.base import History, IntelligenceProviderConfig

async def main():
    # Initialize benchmark
    benchmark = MMLUReduxBenchmark(
        subjects=["high_school_mathematics"],
        filter_error_types=["ok"],
        max_questions_per_subject=10
    )
    
    # Create agent
    agent = Agent(
        name="Math-Evaluator",
        system_prompt="You are an expert at math questions.",
        history=History(),
        ip_config=IntelligenceProviderConfig(
            provider_name="openai",
            version="gpt-4o"
        )
    )
    
    # Run evaluation
    results = await benchmark.evaluate_agent(agent, max_questions=10)
    benchmark.print_results(results)

asyncio.run(main())
```

## 📚 API Reference

### Agent

```python
class Agent(BaseAgent):
    def __init__(
        self,
        name: str,
        system_prompt: str,
        history: History,
        ip_config: IntelligenceProviderConfig,
        tools: List[ToolDefinition] = None,
        enable_parallel_execution: bool = True,
        max_invocations_count: Optional[int] = None,
        fallback_ip_configs: Optional[List[IntelligenceProviderConfig]] = None
    )
    
    async def ask(
        self,
        prompt: str,
        gbs: Optional[GenerationBehaviorSettings] = None
    ) -> Message
    
    async def fork(self, keep_history: bool = True) -> "Agent"
```

### Orchestrator

```python
class Orchestrator:
    def __init__(
        self,
        name: str = "Orchestrator",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        additional_prompt: str = ""
    )
    
    def register_agent(self, agent_type: str, agent: BaseExpertAgent) -> None
    
    async def execute(
        self,
        task: str,
        gbs: Optional[GenerationBehaviorSettings] = None
    ) -> Message
    
    def list_agents(self) -> List[str]
```

### BaseExpertAgent

```python
class BaseExpertAgent(ABC):
    @abstractmethod
    async def solve(
        self,
        problem: str,
        gbs: Optional[GenerationBehaviorSettings] = None
    ) -> Message
    
    def clear_history(self) -> None
    
    @property
    def description(self) -> str
```

## 🧪 Running Tests

```bash

# Run benchmarks
poetry run python src/benchmarks/mmlu_redux.py
poetry run python src/benchmarks/hle.py
```

## 📖 Documentation

- **HLE Benchmark Analysis**: See `src/benchmarks/HLE_ANALYSIS.md`
- **HLE Usage Guide**: See `src/benchmarks/HLE_README.md`
- **Delegation Guide**: See `DELEGATION.md`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- **MMLU-Redux**: Edinburgh DAWG - https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
- Model providers: OpenAI, Anthropic, Google

## 📞 Contact

For questions or feedback, please open an issue on GitHub.
