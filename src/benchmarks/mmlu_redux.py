"""
MMLU-Redux Benchmark Integration

This module provides integration with the MMLU-Redux benchmark dataset from
Edinburgh DAWG. MMLU-Redux is a manually re-annotated subset of 3,000 questions
across 30 MMLU subjects with corrected annotations and error classifications.

Dataset: https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
Paper: https://arxiv.org/abs/2406.04127
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from datasets import load_dataset
from pydantic import BaseModel

from agents.core.simple import SimpleAgent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig
from agents.utils.logs.config import logger
from orchestration import CodeAgent, MathAgent, Orchestrator, ScienceAgent


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

    async def evaluate_agent(
        self,
        agent: SimpleAgent,
        gbs: Optional[GenerationBehaviorSettings] = None,
        max_questions: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Evaluate an agent on the loaded questions

        Args:
            agent: The SimpleAgent to evaluate
            gbs: Optional generation behavior settings
            max_questions: Maximum total questions to evaluate (for quick testing)

        Returns:
            BenchmarkResults containing overall and per-subject accuracy
        """
        if not self.questions:
            self.load_dataset()

        questions_to_eval = self.questions[:max_questions] if max_questions else self.questions
        results: List[MMLUResult] = []

        logger.info(f"Starting evaluation on {len(questions_to_eval)} questions...")

        for i, question in enumerate(questions_to_eval):
            logger.info(f"Evaluating question {i + 1}/{len(questions_to_eval)} - Subject: {question.subject}")

            # Format and ask the question
            prompt = self._format_question(question)

            try:
                # Fork agent to avoid history contamination
                eval_agent = await agent.fork(keep_history=False)
                response = await eval_agent.answer_to(prompt, gbs)

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

                results.append(result)

                logger.info(
                    f"Question {i + 1}: {'✓ Correct' if is_correct else '✗ Incorrect'} "
                    f"(Predicted: {chr(65 + predicted_idx) if predicted_idx >= 0 else 'N/A'}, "
                    f"Correct: {chr(65 + correct_idx)})"
                )

            except Exception as e:
                logger.error(f"Error evaluating question {i + 1}: {e}")
                results.append(
                    MMLUResult(
                        question=question,
                        predicted_answer=-1,
                        predicted_text="ERROR",
                        is_correct=False,
                        response_message=str(e),
                    )
                )

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

        return BenchmarkResults(
            total_questions=total,
            correct_answers=correct,
            accuracy=accuracy,
            results_by_subject=by_subject,
            results_by_error_type=by_error,
            individual_results=results,
        )

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


async def run_benchmark_with_orchestrator(
    max_questions: Optional[int] = None,
    max_questions_per_subject: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    dataset_version: str = "2.0",
):
    """
    Run MMLU-Redux benchmark using the Orchestrator with registered specialized agents.

    Args:
        max_questions: Maximum total questions to evaluate (for quick testing)
        max_questions_per_subject: Maximum questions per subject
        subjects: List of subjects to evaluate on
        dataset_version: Version of MMLU-Redux to use ("1.0" or "2.0", default: "2.0")

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

    # Create orchestrator with specialized agents
    orchestrator = Orchestrator(
        name="MMLU-Orchestrator",
        additional_prompt=f"You are an expert at answering multiple choice questions across various subjects. Current date: {datetime.today().strftime('%Y/%m/%d')}",
    )

    # Register specialized agents
    math_agent = MathAgent(name="MMLU-Math-Expert")
    science_agent = ScienceAgent(name="MMLU-Science-Expert")
    code_agent = CodeAgent(name="MMLU-Code-Expert")

    orchestrator.register_agent("math", math_agent)
    orchestrator.register_agent("science", science_agent)
    orchestrator.register_agent("code", code_agent)

    logger.info(f"Registered agents: {orchestrator.list_agents()}")

    # Warm up
    warmup_response = await orchestrator.execute("Hello!")
    logger.info(f"Warmup response: {warmup_response.content[:100]}...")

    # Run evaluation
    results = await benchmark.evaluate_agent(orchestrator.agent, max_questions=max_questions)

    # Print results
    benchmark.print_results(results)

    return results


async def main(
    max_questions: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    max_questions_per_subject: Optional[int] = None,
    dataset_version: str = "2.0",
):
    """Example usage of MMLU-Redux benchmark"""
    from datetime import datetime

    # Initialize the benchmark (only correct questions)
    benchmark = MMLUReduxBenchmark(
        filter_error_types=["ok"],
        max_questions_per_subject=max_questions_per_subject,
        subjects=subjects,
        dataset_version=dataset_version,
    )

    # Create an agent
    ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

    agent = SimpleAgent(
        name="MMLU-Evaluator",
        system_prompt=f"You are an expert at answering multiple choice questions. Current date: {datetime.today().strftime('%Y/%m/%d')}",
        history=History(),
        ip_config=ip_config,
    )
    print(await agent.answer_to("Hello!"))  # Warm up

    # Run evaluation
    # results = await benchmark.evaluate_agent(agent, max_questions=300)
    results = await benchmark.evaluate_agent(agent, max_questions=max_questions)

    # Print results
    benchmark.print_results(results)


if __name__ == "__main__":
    # Run with simple agent
    # asyncio.run(main())
    # asyncio.run(
    #     main(
    #         max_questions=100,  # Limit for testing
    #         subjects=["college_mathematics", "high_school_mathematics"],  # Specific subjects
    #     )
    # )

    # Run with orchestrator (recommended)
    asyncio.run(
        run_benchmark_with_orchestrator(
            # subjects=["college_mathematics", "high_school_mathematics"],  # Specific subjects
        )
    )
