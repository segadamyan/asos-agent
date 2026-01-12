"""
Tests for HLE benchmark.
"""

import pytest

from benchmarks.hle import HLEBenchmark, run_benchmark_with_orchestrator


@pytest.mark.asyncio
async def test_dataset_loading():
    """Test loading and filtering the HLE dataset."""
    # Test loading with filters
    benchmark = HLEBenchmark(
        categories=["Math", "Computer Science/AI"],
        max_questions_per_category=5,
        skip_images=True,
    )

    benchmark.load_dataset()

    assert len(benchmark.questions) > 0
    assert all(q.category in ["Math", "Computer Science/AI"] for q in benchmark.questions)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_hle_with_orchestrator(check_openai_api_key):
    """Test HLE benchmark with orchestrator."""
    results = await run_benchmark_with_orchestrator(
        max_questions=3,
        skip_images=True,
    )

    assert results is not None
    assert results.total_questions > 0
