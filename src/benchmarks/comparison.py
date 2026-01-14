"""
Baseline vs Tool-Augmented Comparison Framework

Provides side-by-side comparison of baseline LLM agents vs tool-augmented agents.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from agents.core.agent import Agent
from agents.providers.models.base import GenerationBehaviorSettings
from agents.utils.logs.config import logger
from benchmarks.hle import BenchmarkResults as HLEBenchmarkResults
from benchmarks.hle import HLEBenchmark
from benchmarks.mmlu_redux import MMLUReduxBenchmark
from orchestration import Orchestrator


@dataclass
class ComparisonResult:
    """Results of comparing baseline vs tool-augmented agents"""

    baseline_results: Dict  # Baseline benchmark results
    tool_augmented_results: Dict  # Tool-augmented benchmark results
    accuracy_improvement: float  # Absolute improvement in accuracy
    accuracy_improvement_percent: float  # Relative improvement percentage
    latency_overhead: float  # Additional latency from tool usage
    cost_overhead: float  # Additional cost from tool usage
    tool_call_efficiency: float  # Accuracy per tool call
    comparison_metadata: Dict = field(default_factory=dict)


class BaselineComparison:
    """Compare baseline agents vs tool-augmented agents"""

    def __init__(
        self,
        benchmark_name: str,
        baseline_agent: Agent,
        tool_augmented_agent: Agent,
        orchestrator: Optional[Orchestrator] = None,
    ):
        """
        Initialize comparison framework.

        Args:
            benchmark_name: Name of benchmark ('hle' or 'mmlu_redux')
            baseline_agent: Baseline agent (no tools or minimal tools)
            tool_augmented_agent: Tool-augmented agent (with tools/orchestration)
            orchestrator: Optional orchestrator for tool-augmented evaluation
        """
        self.benchmark_name = benchmark_name.lower()
        self.baseline_agent = baseline_agent
        self.tool_augmented_agent = tool_augmented_agent
        self.orchestrator = orchestrator

        if self.benchmark_name == "hle":
            self.benchmark_class = HLEBenchmark
        elif self.benchmark_name == "mmlu_redux":
            self.benchmark_class = MMLUReduxBenchmark
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Use 'hle' or 'mmlu_redux'")

    async def run_comparison(
        self,
        max_questions: Optional[int] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        parallel: bool = False,
        max_concurrent: int = 7,
        save_results: bool = True,
        **benchmark_kwargs,
    ) -> ComparisonResult:
        """
        Run comparison between baseline and tool-augmented agents.

        Args:
            max_questions: Maximum questions to evaluate
            gbs: Generation behavior settings
            parallel: Whether to run in parallel
            max_concurrent: Max concurrent requests
            save_results: Whether to save comparison results
            **benchmark_kwargs: Additional arguments for benchmark initialization

        Returns:
            ComparisonResult with detailed comparison metrics
        """
        logger.info("Starting comparison: Baseline vs Tool-Augmented on %s", self.benchmark_name)

        # Initialize benchmark
        benchmark = self.benchmark_class(**benchmark_kwargs)

        # Run baseline evaluation
        logger.info("Evaluating baseline agent...")
        baseline_results = await benchmark.evaluate_agent(
            self.baseline_agent,
            gbs=gbs,
            max_questions=max_questions,
            parallel=parallel,
            max_concurrent=max_concurrent,
            save_results=True,  # We'll save combined results
        )

        # Run tool-augmented evaluation
        logger.info("Evaluating tool-augmented agent...")
        if self.orchestrator:
            # Use orchestrator factory if provided
            def create_orchestrator():
                return self.orchestrator

            tool_augmented_results = await benchmark.evaluate_agent(
                self.tool_augmented_agent,
                gbs=gbs,
                max_questions=max_questions,
                parallel=parallel,
                max_concurrent=max_concurrent,
                save_results=True,
                orchestrator_factory=create_orchestrator if self.orchestrator else None,
            )
        else:
            tool_augmented_results = await benchmark.evaluate_agent(
                self.tool_augmented_agent,
                gbs=gbs,
                max_questions=max_questions,
                parallel=parallel,
                max_concurrent=max_concurrent,
                save_results=True,
            )

        # Calculate comparison metrics
        comparison = self._calculate_comparison(baseline_results, tool_augmented_results)

        # Save results if requested
        if save_results:
            self._save_comparison(comparison, baseline_results, tool_augmented_results)

        return comparison

    def _calculate_comparison(
        self,
        baseline_results: HLEBenchmarkResults,
        tool_augmented_results: HLEBenchmarkResults,
    ) -> ComparisonResult:
        """Calculate comparison metrics between baseline and tool-augmented results"""

        # Accuracy improvement
        accuracy_improvement = tool_augmented_results.accuracy - baseline_results.accuracy
        accuracy_improvement_percent = (
            (accuracy_improvement / baseline_results.accuracy * 100) if baseline_results.accuracy > 0 else 0.0
        )

        # Latency overhead
        latency_overhead = tool_augmented_results.average_latency_seconds - baseline_results.average_latency_seconds

        # Cost overhead
        cost_overhead = tool_augmented_results.total_cost_usd - baseline_results.total_cost_usd

        # Tool call efficiency (accuracy per tool call)
        tool_call_efficiency = (
            tool_augmented_results.accuracy / tool_augmented_results.average_tool_calls_per_question
            if tool_augmented_results.average_tool_calls_per_question > 0
            else 0.0
        )

        # Metadata
        comparison_metadata = {
            "baseline_accuracy": baseline_results.accuracy,
            "tool_augmented_accuracy": tool_augmented_results.accuracy,
            "baseline_latency": baseline_results.average_latency_seconds,
            "tool_augmented_latency": tool_augmented_results.average_latency_seconds,
            "baseline_tool_calls": baseline_results.total_tool_calls,
            "tool_augmented_tool_calls": tool_augmented_results.total_tool_calls,
            "baseline_cost": baseline_results.total_cost_usd,
            "tool_augmented_cost": tool_augmented_results.total_cost_usd,
        }

        return ComparisonResult(
            baseline_results=self._results_to_dict(baseline_results),
            tool_augmented_results=self._results_to_dict(tool_augmented_results),
            accuracy_improvement=accuracy_improvement,
            accuracy_improvement_percent=accuracy_improvement_percent,
            latency_overhead=latency_overhead,
            cost_overhead=cost_overhead,
            tool_call_efficiency=tool_call_efficiency,
            comparison_metadata=comparison_metadata,
        )

    def _results_to_dict(self, results) -> Dict:
        """Convert benchmark results to dictionary for serialization"""
        return {
            "accuracy": results.accuracy,
            "total_questions": results.total_questions,
            "correct_answers": results.correct_answers,
            "average_latency_seconds": results.average_latency_seconds,
            "total_tool_calls": results.total_tool_calls,
            "average_tool_calls_per_question": results.average_tool_calls_per_question,
            "total_cost_usd": results.total_cost_usd,
            "error_breakdown": results.error_breakdown,
        }

    def _save_comparison(
        self,
        comparison: ComparisonResult,
        _baseline_results: HLEBenchmarkResults,
        _tool_augmented_results: HLEBenchmarkResults,
    ):
        """Save comparison results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.benchmark_name}_comparison_{timestamp}.json"

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "benchmark": self.benchmark_name,
                "comparison_type": "baseline_vs_tool_augmented",
            },
            "comparison_metrics": {
                "accuracy_improvement": comparison.accuracy_improvement,
                "accuracy_improvement_percent": comparison.accuracy_improvement_percent,
                "latency_overhead_seconds": comparison.latency_overhead,
                "cost_overhead_usd": comparison.cost_overhead,
                "tool_call_efficiency": comparison.tool_call_efficiency,
            },
            "detailed_metrics": comparison.comparison_metadata,
            "baseline_results": comparison.baseline_results,
            "tool_augmented_results": comparison.tool_augmented_results,
        }

        output_file = Path(output_path)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Comparison results saved to: %s", output_file.absolute())
        print(f"\n📊 Comparison results saved to: {output_file.absolute()}")

    def print_comparison(self, comparison: ComparisonResult):
        """Pretty print comparison results"""
        print("\n" + "=" * 80)
        print(f"Baseline vs Tool-Augmented Comparison ({self.benchmark_name.upper()})")
        print("=" * 80)

        print(f"\n{'Metric':<40} {'Baseline':<20} {'Tool-Augmented':<20} {'Difference':<20}")
        print("-" * 80)

        baseline_acc = comparison.comparison_metadata["baseline_accuracy"]
        tool_acc = comparison.comparison_metadata["tool_augmented_accuracy"]
        print(f"{'Accuracy':<40} {baseline_acc:>19.2%} {tool_acc:>19.2%} {comparison.accuracy_improvement:>+19.2%}")

        baseline_lat = comparison.comparison_metadata["baseline_latency"]
        tool_lat = comparison.comparison_metadata["tool_augmented_latency"]
        print(
            f"{'Avg Latency (seconds)':<40} {baseline_lat:>19.2f} {tool_lat:>19.2f} {comparison.latency_overhead:>+19.2f}"
        )

        baseline_cost = comparison.comparison_metadata["baseline_cost"]
        tool_cost = comparison.comparison_metadata["tool_augmented_cost"]
        print(
            f"{'Total Cost (USD)':<40} ${baseline_cost:>18.4f} ${tool_cost:>18.4f} ${comparison.cost_overhead:>+18.4f}"
        )

        baseline_tools = comparison.comparison_metadata["baseline_tool_calls"]
        tool_tools = comparison.comparison_metadata["tool_augmented_tool_calls"]
        print(f"{'Total Tool Calls':<40} {baseline_tools:>19} {tool_tools:>19} {tool_tools - baseline_tools:>+19}")

        print("\n" + "-" * 80)
        print(
            f"Accuracy Improvement: {comparison.accuracy_improvement:+.2%} ({comparison.accuracy_improvement_percent:+.2f}%)"
        )
        print(f"Tool Call Efficiency: {comparison.tool_call_efficiency:.4f} accuracy per tool call")
        print("=" * 80)


async def compare_baseline_vs_orchestrator(
    benchmark_name: str,
    baseline_agent: Agent,
    orchestrator: Orchestrator,
    max_questions: Optional[int] = None,
    **kwargs,
) -> ComparisonResult:
    """
    Convenience function to compare baseline agent vs orchestrator.

    Args:
        benchmark_name: 'hle' or 'mmlu_redux'
        baseline_agent: Baseline agent without tools
        orchestrator: Orchestrator with expert agents
        max_questions: Maximum questions to evaluate
        **kwargs: Additional arguments for benchmark and comparison

    Returns:
        ComparisonResult
    """
    comparison = BaselineComparison(
        benchmark_name=benchmark_name,
        baseline_agent=baseline_agent,
        tool_augmented_agent=orchestrator.agent,
        orchestrator=orchestrator,
    )

    result = await comparison.run_comparison(max_questions=max_questions, **kwargs)
    comparison.print_comparison(result)

    return result
