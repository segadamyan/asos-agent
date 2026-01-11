"""
MMLU-Redux Benchmark Integration

This module provides integration with the MMLU-Redux benchmark dataset from
Edinburgh DAWG. MMLU-Redux is a manually re-annotated subset of 3,000 questions
across 30 MMLU subjects with corrected annotations and error classifications.

Dataset: https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
Paper: https://arxiv.org/abs/2406.04127
"""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from pydantic import BaseModel

from agents.core.agent import Agent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig
from agents.utils.logs.config import logger
from orchestration import BusinessLawAgent, CodeAgent, MathAgent, Orchestrator, ScienceAgent


class MMLUQuestion(BaseModel):
    """Represents a single MMLU-Redux question"""

    question: str
    choices: List[str]
    answer: int
    error_type: str
    source: Optional[str] = None
    correct_answer: Optional[int] = None
    potential_reason: Optional[str] = None
    subject: str
    index: int


class MMLUResult(BaseModel):
    """Result of evaluating a single question"""

    question: MMLUQuestion
    predicted_answer: int
    predicted_text: str
    is_correct: bool
    response_message: str


@dataclass
class BenchmarkResults:
    """Overall benchmark results"""

    total_questions: int
    correct_answers: int
    accuracy: float
    results_by_subject: Dict[str, Dict[str, float]]
    results_by_error_type: Dict[str, Dict[str, float]]
    individual_results: List[MMLUResult]


class MMLUReduxBenchmark:
    """
    MMLU-Redux Benchmark Runner

    This class handles loading the MMLU-Redux dataset and evaluating agents on it.
    """

    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        filter_error_types: Optional[List[str]] = None,
        max_questions_per_subject: Optional[int] = None,
        dataset_version: str = "1.0",
    ):
        """
        Initialize the benchmark

        Args:
            subjects: List of subjects to evaluate on. If None, uses all 30 subjects.
            filter_error_types: Only include questions with these error types.
                               Use ["ok"] to only evaluate on correct questions.
            max_questions_per_subject: Maximum number of questions per subject to evaluate
            dataset_version: Version of MMLU-Redux to use ("1.0" or "2.0")
        """
        self.subjects = subjects
        self.filter_error_types = filter_error_types or ["ok"]
        self.max_questions_per_subject = max_questions_per_subject
        self.dataset_version = dataset_version
        self.questions: List[MMLUQuestion] = []

    def load_dataset(self):
        """Load the MMLU-Redux dataset from HuggingFace"""
        dataset_name = "edinburgh-dawg/mmlu-redux-2.0" if self.dataset_version == "2.0" else "edinburgh-dawg/mmlu-redux"
        logger.info(f"Loading MMLU-Redux dataset (version {self.dataset_version}) from {dataset_name}...")

        # Available subjects in MMLU-Redux
        available_subjects = [
            "anatomy",
            "business_ethics",
            "clinical_knowledge",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "econometrics",
            "electrical_engineering",
            "formal_logic",
            "global_facts",
            "high_school_chemistry",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "human_aging",
            "logical_fallacies",
            "machine_learning",
            "miscellaneous",
            "philosophy",
            "professional_accounting",
            "public_relations",
            "virology",
            "conceptual_physics",
            "high_school_us_history",
            "astronomy",
            "high_school_geography",
            "high_school_macroeconomics",
            "professional_law",
        ]

        # Determine which subjects to load
        subjects_to_load = self.subjects if self.subjects else available_subjects

        for subject in subjects_to_load:
            logger.info(f"Loading subject: {subject}")
            try:
                # Load dataset for this subject
                subject_data = load_dataset(dataset_name, subject, split="test")

                # Filter by error type if specified
                if self.filter_error_types:
                    subject_data = subject_data.filter(lambda x: x["error_type"] in self.filter_error_types)

                count = 0
                for idx, item in enumerate(subject_data):
                    if self.max_questions_per_subject and count >= self.max_questions_per_subject:
                        break

                    self.questions.append(
                        MMLUQuestion(
                            question=item["question"],
                            choices=item["choices"],
                            answer=item["answer"],
                            error_type=item["error_type"],
                            source=item.get("source"),
                            correct_answer=item.get("correct_answer"),
                            potential_reason=item.get("potential_reason"),
                            subject=subject,
                            index=idx,
                        )
                    )
                    count += 1

                logger.info(f"Loaded {count} questions from {subject}")

            except Exception as e:
                logger.error(f"Error loading subject {subject}: {e}")

        logger.info(f"Total loaded: {len(self.questions)} questions from MMLU-Redux")

    def _format_question(self, question: MMLUQuestion) -> str:
        """Format a question for the agent"""
        choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(question.choices)])

        prompt = f"""Answer the following multiple choice question. Provide ONLY the letter (A, B, C, or D) of the correct answer.

Question: {question.question}

Choices:
{choices_text}

Provide ONLY the letter (A, B, C, or D) of the correct answer. You may include a brief explanation, but the answer letter must be clear.

Answer:"""

        return prompt

    def _extract_answer(self, response: str) -> int:
        """Extract the answer index from the agent's response"""
        response = response.strip().upper()

        # Look for explicit answer patterns like "Answer: B" or "Correct Answer: B" or "**B.**"

        # Pattern 1: "Answer: X" or "Answer is X" or "Answer X"
        match = re.search(r"(?:ANSWER|CHOICE)(?:\s+IS)?[\s:]+([A-D])", response)
        if match:
            return ord(match.group(1)) - 65

        # Pattern 2: "Correct Answer: **B.**" or similar
        match = re.search(r"CORRECT\s+ANSWER[\s:]+\*?\*?([A-D])", response)
        if match:
            return ord(match.group(1)) - 65

        # Pattern 3: Look for standalone letter near the start (first 100 chars)
        for letter in ["A", "B", "C", "D"]:
            # Match letter that's not part of a word (standalone or with punctuation)
            match = re.search(rf"\b{letter}\b", response[:100])
            if match:
                return ord(letter) - 65

        # Pattern 4: Look for letter followed by period or colon
        for letter in ["A", "B", "C", "D"]:
            if f"{letter}." in response[:100] or f"{letter}:" in response[:100]:
                return ord(letter) - 65

        # Fallback: First occurrence of A, B, C, or D as whole word
        for letter in ["A", "B", "C", "D"]:
            match = re.search(rf"\b{letter}\b", response)
            if match:
                return ord(letter) - 65

        # If no clear answer, return -1
        logger.warning(f"Could not extract answer from response: {response[:100]}")
        return -1

    async def _evaluate_question(
        self,
        agent: Agent,
        question: MMLUQuestion,
        question_num: int,
        total_questions: int,
        gbs: Optional[GenerationBehaviorSettings] = None,
        orchestrator_factory=None,
    ) -> MMLUResult:
        """Evaluate a single question

        Args:
            agent: The Agent to evaluate
            question: The question to evaluate
            question_num: Question number (1-indexed for logging)
            total_questions: Total number of questions being evaluated
            gbs: Optional generation behavior settings
            orchestrator_factory: Optional function that creates a new Orchestrator for each question

        Returns:
            MMLUResult for this question
        """
        logger.info(f"Evaluating question {question_num}/{total_questions} - Subject: {question.subject}")

        prompt = self._format_question(question)

        try:
            # Create fresh orchestrator for this question if factory is provided
            # This ensures complete isolation and prevents context overflow
            if orchestrator_factory:
                fresh_orchestrator = orchestrator_factory()
                eval_agent = await fresh_orchestrator.agent.fork(keep_history=False)
            else:
                # Fork agent to avoid history contamination
                eval_agent = await agent.fork(keep_history=False)

            response = await eval_agent.ask(prompt, gbs)

            # Extract predicted answer
            predicted_idx = self._extract_answer(response.content)
            predicted_text = question.choices[predicted_idx] if 0 <= predicted_idx < 4 else "INVALID"

            # Use correct_answer if available, otherwise use answer
            correct_idx = question.correct_answer if question.correct_answer is not None else question.answer

            is_correct = predicted_idx == correct_idx

            result = MMLUResult(
                question=question,
                predicted_answer=predicted_idx,
                predicted_text=predicted_text,
                is_correct=is_correct,
                response_message=response.content,
            )

            logger.info(
                f"Question {question_num}: {'✓ Correct' if is_correct else '✗ Incorrect'} "
                f"(Predicted: {chr(65 + predicted_idx) if predicted_idx >= 0 else 'N/A'}, "
                f"Correct: {chr(65 + correct_idx)})"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating question {question_num}: {e}")
            return MMLUResult(
                question=question,
                predicted_answer=-1,
                predicted_text="ERROR",
                is_correct=False,
                response_message=str(e),
            )

    async def evaluate_agent(
        self,
        agent: Agent,
        gbs: Optional[GenerationBehaviorSettings] = None,
        max_questions: Optional[int] = None,
        parallel: bool = False,
        max_concurrent: int = 10,
        save_results: bool = False,
        output_path: Optional[str] = None,
        orchestrator_factory=None,
    ) -> BenchmarkResults:
        """
        Evaluate an agent on the loaded questions

        Args:
            agent: The Agent to evaluate (can be orchestrator.agent if using orchestrator)
            gbs: Optional generation behavior settings
            max_questions: Maximum total questions to evaluate (for quick testing)
            parallel: If True, evaluate questions in parallel using asyncio.gather().
                     If False (default), evaluate sequentially to avoid rate limits.
            max_concurrent: Maximum number of concurrent requests when parallel=True.
                           Default is 10. Adjust based on your API rate limits.
            save_results: If True, automatically save results to JSON file
            output_path: Custom path for JSON output (only used if save_results=True)
            orchestrator_factory: Optional function that creates fresh Orchestrator instances.
                                Use when agent needs orchestration with expert agents.

        Returns:
            BenchmarkResults containing overall and per-subject accuracy
        """
        if not self.questions:
            self.load_dataset()

        questions_to_eval = self.questions[:max_questions] if max_questions else self.questions

        logger.info(
            f"Starting evaluation on {len(questions_to_eval)} questions..."
            f" (mode: {'parallel' if parallel else 'sequential'}"
            f"{f', max_concurrent: {max_concurrent}' if parallel else ''})"
        )

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def evaluate_with_semaphore(question, question_num):
                async with semaphore:
                    return await self._evaluate_question(
                        agent=agent,
                        question=question,
                        question_num=question_num,
                        total_questions=len(questions_to_eval),
                        gbs=gbs,
                        orchestrator_factory=orchestrator_factory,
                    )

            tasks = [evaluate_with_semaphore(question, i + 1) for i, question in enumerate(questions_to_eval)]
            results = await asyncio.gather(*tasks)
            results = list(results)  # Convert tuple to list
        else:
            # Sequential evaluation (default, safer for rate limits)
            results: List[MMLUResult] = []
            for i, question in enumerate(questions_to_eval):
                result = await self._evaluate_question(
                    agent=agent,
                    question=question,
                    question_num=i + 1,
                    total_questions=len(questions_to_eval),
                    gbs=gbs,
                    orchestrator_factory=orchestrator_factory,
                )
                results.append(result)

        # Calculate overall metrics
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-subject metrics
        by_subject = {}
        for result in results:
            subject = result.question.subject
            if subject not in by_subject:
                by_subject[subject] = {"total": 0, "correct": 0, "accuracy": 0.0}

            by_subject[subject]["total"] += 1
            if result.is_correct:
                by_subject[subject]["correct"] += 1

        for subject in by_subject:
            by_subject[subject]["accuracy"] = by_subject[subject]["correct"] / by_subject[subject]["total"]

        # Calculate per-error-type metrics
        by_error = {}
        for result in results:
            error_type = result.question.error_type
            if error_type not in by_error:
                by_error[error_type] = {"total": 0, "correct": 0, "accuracy": 0.0}

            by_error[error_type]["total"] += 1
            if result.is_correct:
                by_error[error_type]["correct"] += 1

        for error_type in by_error:
            by_error[error_type]["accuracy"] = by_error[error_type]["correct"] / by_error[error_type]["total"]

        benchmark_results = BenchmarkResults(
            total_questions=total,
            correct_answers=correct,
            accuracy=accuracy,
            results_by_subject=by_subject,
            results_by_error_type=by_error,
            individual_results=results,
        )

        # Save results to JSON if requested
        if save_results:
            self.save_results_to_json(benchmark_results, output_path)

        return benchmark_results

    def print_results(self, results: BenchmarkResults):
        """Pretty print the benchmark results"""
        print("\n" + "=" * 80)
        print("MMLU-Redux Benchmark Results")
        print("=" * 80)
        print(f"\nOverall Accuracy: {results.accuracy:.2%} ({results.correct_answers}/{results.total_questions})")

        print("\n" + "-" * 80)
        print("Results by Subject:")
        print("-" * 80)
        for subject, metrics in sorted(results.results_by_subject.items()):
            print(f"{subject:40s} {metrics['accuracy']:6.2%} ({metrics['correct']}/{metrics['total']})")

        print("\n" + "-" * 80)
        print("Results by Error Type:")
        print("-" * 80)
        for error_type, metrics in sorted(results.results_by_error_type.items()):
            print(f"{error_type:30s} {metrics['accuracy']:6.2%} ({metrics['correct']}/{metrics['total']})")

        print("\n" + "=" * 80)

    def save_results_to_json(self, results: BenchmarkResults, output_path: Optional[str] = None):
        """
        Save benchmark results to a JSON file for later analysis

        Args:
            results: The BenchmarkResults to save
            output_path: Path to save the JSON file. If None, generates a timestamped filename.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"mmlu_redux_results_{timestamp}.json"

        # Convert results to a serializable format
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": results.total_questions,
                "correct_answers": results.correct_answers,
                "accuracy": results.accuracy,
            },
            "summary": {
                "by_subject": results.results_by_subject,
                "by_error_type": results.results_by_error_type,
            },
            "individual_results": [
                {
                    "question_text": result.question.question,
                    "choices": result.question.choices,
                    "subject": result.question.subject,
                    "error_type": result.question.error_type,
                    "correct_answer_index": result.question.correct_answer
                    if result.question.correct_answer is not None
                    else result.question.answer,
                    "correct_answer_text": result.question.choices[
                        result.question.correct_answer
                        if result.question.correct_answer is not None
                        else result.question.answer
                    ],
                    "predicted_answer_index": result.predicted_answer,
                    "predicted_answer_text": result.predicted_text,
                    "full_response": result.response_message,
                    "is_correct": result.is_correct,
                }
                for result in results.individual_results
            ],
        }

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file.absolute()}")
        print(f"\n📄 Results saved to: {output_file.absolute()}")


async def run_benchmark_with_orchestrator(
    max_questions: Optional[int] = None,
    max_questions_per_subject: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    dataset_version: str = "2.0",
    parallel: bool = False,
    max_concurrent: int = 10,
):
    """
    Run MMLU-Redux benchmark using the Orchestrator with registered specialized agents.

    Args:
        max_questions: Maximum total questions to evaluate (for quick testing)
        max_questions_per_subject: Maximum questions per subject
        subjects: List of subjects to evaluate on
        dataset_version: Version of MMLU-Redux to use ("1.0" or "2.0", default: "2.0")
        parallel: If True, evaluate questions in parallel
        max_concurrent: Maximum concurrent requests when parallel=True (default: 10)

    Returns:
        BenchmarkResults
    """

    # Initialize the benchmark (only correct questions)
    benchmark = MMLUReduxBenchmark(
        filter_error_types=["ok"],
        max_questions_per_subject=max_questions_per_subject,
        subjects=subjects,
        dataset_version=dataset_version,
    )

    ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

    def create_fresh_orchestrator():
        orch = Orchestrator(
            name="MMLU-Orchestrator",
            additional_prompt=f"You are an expert at answering multiple choice questions across various subjects. Current date: {datetime.today().strftime('%Y/%m/%d')}",
            ip_config=ip_config,
        )

        orch.register_agent("math", MathAgent(name="MMLU-Math-Expert", ip_config=ip_config))
        orch.register_agent("science", ScienceAgent(name="MMLU-Science-Expert", ip_config=ip_config))
        orch.register_agent("code", CodeAgent(name="MMLU-Code-Expert", ip_config=ip_config))
        orch.register_agent("business_law", BusinessLawAgent(name="MMLU-BusinessLaw-Expert", ip_config=ip_config))

        return orch

    warmup_orchestrator = create_fresh_orchestrator()
    logger.info(f"Registered agents: {warmup_orchestrator.list_agents()}")

    # Warm up
    warmup_response = await warmup_orchestrator.execute("Hello!")
    logger.info(f"Warmup response: {warmup_response.content[:100]}...")

    # Run evaluation with factory - each question gets a fresh orchestrator
    # We pass a dummy agent since the factory will create the actual agents
    dummy_agent = Agent(
        name="Dummy",
        system_prompt="",
        history=History(),
        ip_config=ip_config,
    )

    results = await benchmark.evaluate_agent(
        dummy_agent,  # Will be replaced by fresh orchestrators
        max_questions=max_questions,
        parallel=parallel,
        max_concurrent=max_concurrent,
        save_results=True,
        orchestrator_factory=create_fresh_orchestrator,
    )

    # Print results
    benchmark.print_results(results)

    return results


async def main(
    max_questions: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    max_questions_per_subject: Optional[int] = None,
    parallel: bool = True,
    dataset_version: str = "2.0",
):
    """Example usage of MMLU-Redux benchmark"""

    # Initialize the benchmark (only correct questions)
    benchmark = MMLUReduxBenchmark(
        filter_error_types=["ok"],
        max_questions_per_subject=max_questions_per_subject,
        subjects=subjects,
        dataset_version=dataset_version,
    )

    # Create an agent
    ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

    agent = Agent(
        name="MMLU-Evaluator",
        system_prompt=f"You are an expert at answering multiple choice questions. Current date: {datetime.today().strftime('%Y/%m/%d')}",
        history=History(),
        ip_config=ip_config,
    )
    print(await agent.ask("Hello!"))  # Warm up

    # Run evaluation
    # results = await benchmark.evaluate_agent(agent, max_questions=300)
    results = await benchmark.evaluate_agent(agent, max_questions=max_questions, parallel=parallel, save_results=True)

    benchmark.print_results(results)


if __name__ == "__main__":
    # Run with simple agent
    # asyncio.run(main())
    asyncio.run(
        main(
            parallel=True,
            subjects=[
                "philosophy",
                "logical_fallacies",
                "formal_logic",
                "high_school_us_history",
                "high_school_geography",
                "high_school_macroeconomics",
                "political_science",
            ],  # Specific subjects
        )
    )

    # Run with orchestrator (recommended)
    # asyncio.run(
    #     run_benchmark_with_orchestrator(
    #         parallel=True,
    #         subjects=["philosophy", "logical_fallacies", "formal_logic", "high_school_us_history", "high_school_geography", "high_school_macroeconomics", "political_science"],  # Specific subjects
    #     )
    # )

    # Run with orchestrator (recommended)
    asyncio.run(
        run_benchmark_with_orchestrator(
            # subjects=["college_mathematics", "high_school_mathematics"],  # Specific subjects
        )
    )
