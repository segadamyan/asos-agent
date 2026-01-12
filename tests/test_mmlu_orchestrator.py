"""
Tests for MMLU-Redux benchmark with orchestrator.
"""

import pytest

from benchmarks.mmlu_redux import run_benchmark_with_orchestrator


@pytest.mark.asyncio
@pytest.mark.slow
async def test_mmlu_with_orchestrator(check_openai_api_key):
    """Run a quick test with just math subjects."""
    # Test with only math subjects and limited questions
    results = await run_benchmark_with_orchestrator(
        max_questions=5,  # Only 5 total questions for quick test
        subjects=["college_mathematics"],
    )

    assert results is not None
    assert results.total_questions > 0
    assert 0 <= results.accuracy <= 1
