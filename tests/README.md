# Tests

This directory contains all test files for the asos-agent project.

## Running Tests

### Install dependencies first
```bash
poetry install
```

### Run all tests
```bash
poetry run pytest
```

### Run specific test file
```bash
poetry run pytest tests/test_agents.py
```

### Run specific test function
```bash
poetry run pytest tests/test_agents.py::test_simple_agent -v
```

### Run tests excluding slow tests
```bash
poetry run pytest -m "not slow"
```

### Run with coverage
```bash
poetry run pytest --cov=src --cov-report=html
```

## Test Organization

- **test_agents.py** - Tests for orchestration agents (Math, Science, Code, BusinessLaw)
- **test_provider.py** - Tests for provider integration (OpenAI, Anthropic, Google)
- **test_orchestrator.py** - Tests for the orchestrator with tools
- **test_delegation.py** - Tests for delegation system
- **test_tool_calling.py** - Tests for tool calling functionality
- **test_orchestrator_delegate.py** - Tests for orchestrator with built-in delegation
- **test_responses_api.py** - Tests for OpenAI API compatibility
- **test_hle.py** - Tests for HLE benchmark
- **test_mmlu_orchestrator.py** - Tests for MMLU-Redux benchmark
- **test_mcp_math.py** - Tests for MCP math integration

## Environment Variables

Most tests require API keys:
- `OPENAI_API_KEY` - Required for most tests
- `ANTHROPIC_API_KEY` - Optional
- `GOOGLE_API_KEY` - Optional

Set them in your environment:
```bash
export OPENAI_API_KEY='your-key-here'
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.asyncio` - Async test (handled automatically)
- `@pytest.mark.slow` - Slow-running test (benchmarks, integration tests)

Skip slow tests with:
```bash
poetry run pytest -m "not slow"
```
