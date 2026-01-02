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
from agents.core.simple import SimpleAgent
from agents.providers.models.base import History, IntelligenceProviderConfig

async def main():
    # Create a simple agent
    agent = SimpleAgent(
        name="MyAgent",
        system_prompt="You are a helpful AI assistant.",
        history=History(),
        ip_config=IntelligenceProviderConfig(
            provider_name="openai",
            version="gpt-4o"
        )
    )
    
    # Ask a question
    response = await agent.answer_to("What is the capital of France?")
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

```
ASOS Agent Framework
├── Agents Layer
│   ├── SimpleAgent (Core agent with LLM communication)
│   ├── BaseExpertAgent (Abstract class for specialists)
│   └── Tool Execution System
├── Orchestration Layer
│   ├── Orchestrator (Task coordinator)
│   └── Expert Agents (Math, Science, Code)
├── Provider Layer
│   ├── OpenAI Provider
│   ├── Anthropic Provider
│   └── Gemini Provider
└── Benchmarks Layer
    └── MMLU-Redux
```

## 🧩 Core Components

### 1. SimpleAgent

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
        return await self.agent.answer_to(problem, gbs=effective_gbs)
    
    @property
    def description(self) -> str:
        return f"{self.name}: Expert in custom domain"
```

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
from agents.core.simple import SimpleAgent

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

agent = SimpleAgent(
    name="WeatherAgent",
    system_prompt="You help users with weather information.",
    history=History(),
    ip_config=IntelligenceProviderConfig(provider_name="openai", version="gpt-4o"),
    tools=[weather_tool]
)

response = await agent.answer_to("What's the weather in Paris?")
```

### Example 3: Benchmark Evaluation

```python
import asyncio
from src.benchmarks.mmlu_redux import MMLUReduxBenchmark
from agents.core.simple import SimpleAgent
from agents.providers.models.base import History, IntelligenceProviderConfig

async def main():
    # Initialize benchmark
    benchmark = MMLUReduxBenchmark(
        subjects=["high_school_mathematics"],
        filter_error_types=["ok"],
        max_questions_per_subject=10
    )
    
    # Create agent
    agent = SimpleAgent(
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

### SimpleAgent

```python
class SimpleAgent(BaseAgent):
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
    
    async def answer_to(
        self,
        prompt: str,
        gbs: Optional[GenerationBehaviorSettings] = None
    ) -> Message
    
    async def fork(self, keep_history: bool = True) -> "SimpleAgent"
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
