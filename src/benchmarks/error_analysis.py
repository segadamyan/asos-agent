"""
Error Analysis Module

Provides systematic categorization of errors in benchmark evaluations.
Analyzes different error modes: hallucination, tool misuse, context mistakes, logic failures, etc.
"""

import re
from enum import Enum


class ErrorCategory(str, Enum):
    """Categories of errors in agent responses"""

    CORRECT = "correct"
    HALLUCINATION = "hallucination"  # Model generates plausible but incorrect information
    TOOL_MISUSE = "tool_misuse"  # Incorrect tool usage or parameters
    CONTEXT_MISTAKE = "context_mistake"  # Context window or memory issues
    LOGIC_FAILURE = "logic_failure"  # Incorrect reasoning or calculation
    EXTRACTION_ERROR = "extraction_error"  # Answer extraction failed
    FORMAT_ERROR = "format_error"  # Wrong format (e.g., letter vs number)
    PARTIAL_CORRECT = "partial_correct"  # Partially correct but incomplete
    TIMEOUT = "timeout"  # Request timed out
    PROVIDER_ERROR = "provider_error"  # Provider API error
    UNKNOWN = "unknown"  # Unclassified error


class ErrorAnalyzer:
    """Analyzes agent responses to categorize errors"""

    @staticmethod
    def categorize_error(
        predicted_answer: str,
        correct_answer: str,
        response_message: str,
        is_correct: bool,
        tool_call_count: int = 0,
        answer_type: str = "exactMatch",
    ) -> ErrorCategory:
        """
        Categorize the error type for an incorrect answer.

        Args:
            predicted_answer: The extracted predicted answer
            correct_answer: The correct answer
            response_message: Full response message from agent
            is_correct: Whether the answer is correct
            tool_call_count: Number of tool calls made
            answer_type: Type of answer (exactMatch or multipleChoice)

        Returns:
            ErrorCategory enum value
        """
        if is_correct:
            return ErrorCategory.CORRECT

        # Check for timeout or provider errors
        if "timeout" in response_message.lower() or "timed out" in response_message.lower():
            return ErrorCategory.TIMEOUT

        if "error" in response_message.lower() and "provider" in response_message.lower():
            return ErrorCategory.PROVIDER_ERROR

        # Check for extraction errors (predicted answer is ERROR or invalid)
        if predicted_answer.upper() in ["ERROR", "INVALID", "N/A", ""]:
            return ErrorCategory.EXTRACTION_ERROR

        # For multiple choice, check format errors
        if answer_type == "multipleChoice":
            if not re.match(r"^[A-J]$", predicted_answer.strip().upper()):
                return ErrorCategory.FORMAT_ERROR

        # Check for tool misuse patterns
        if tool_call_count > 0:
            # Look for tool error messages
            if any(
                keyword in response_message.lower()
                for keyword in [
                    "tool error",
                    "tool failed",
                    "invalid tool",
                    "tool execution failed",
                    "cannot execute",
                ]
            ):
                return ErrorCategory.TOOL_MISUSE

        # Check for context/memory mistakes
        if any(
            keyword in response_message.lower()
            for keyword in [
                "context overflow",
                "token limit",
                "too long",
                "memory",
                "forgot",
                "lost context",
            ]
        ):
            return ErrorCategory.CONTEXT_MISTAKE

        # Check for logic failures (common patterns)
        logic_failure_patterns = [
            r"incorrect.*calculation",
            r"wrong.*formula",
            r"miscalculated",
            r"logic error",
            r"reasoning.*incorrect",
        ]
        if any(re.search(pattern, response_message.lower()) for pattern in logic_failure_patterns):
            return ErrorCategory.LOGIC_FAILURE

        # Check for partial correctness
        # If predicted answer contains parts of correct answer but is incomplete
        if answer_type == "exactMatch":
            correct_lower = correct_answer.lower()
            predicted_lower = predicted_answer.lower()
            if correct_lower in predicted_lower or predicted_lower in correct_lower:
                if len(predicted_answer) < len(correct_answer) * 0.7:  # Significantly shorter
                    return ErrorCategory.PARTIAL_CORRECT

        # Check for hallucination patterns
        # Hallucination: model generates plausible but incorrect information
        hallucination_indicators = [
            len(response_message) > 500,  # Very verbose responses
            "i believe" in response_message.lower(),
            "likely" in response_message.lower(),
            "probably" in response_message.lower(),
            "might be" in response_message.lower(),
        ]
        if sum(hallucination_indicators) >= 2:
            # Check if response contains made-up facts or numbers
            if ErrorAnalyzer._contains_hallucinated_content(response_message, correct_answer):
                return ErrorCategory.HALLUCINATION

        # Default to unknown if no specific pattern matches
        return ErrorCategory.UNKNOWN

    @staticmethod
    def _contains_hallucinated_content(response: str, correct_answer: str) -> bool:
        """
        Check if response contains hallucinated content.

        Args:
            response: Full response message
            correct_answer: Correct answer for comparison

        Returns:
            True if response likely contains hallucinated content
        """
        # Look for patterns that suggest hallucination
        # Very specific numbers or facts that don't match correct answer
        # Excessive confidence in wrong answer
        confidence_phrases = [
            "definitely",
            "certainly",
            "absolutely",
            "without a doubt",
            "clearly",
        ]
        if any(phrase in response.lower() for phrase in confidence_phrases):
            # If high confidence but wrong, likely hallucination
            return True

        # Check for made-up citations or references
        citation_patterns = [
            r"according to [A-Z][a-z]+",
            r"studies show",
            r"research indicates",
            r"experts say",
        ]
        if any(re.search(pattern, response, re.IGNORECASE) for pattern in citation_patterns):
            return True

        return False

    @staticmethod
    def analyze_error_distribution(results: list) -> dict:
        """
        Analyze error distribution across all results.

        Args:
            results: List of result objects (HLEResult or MMLUResult)

        Returns:
            Dictionary mapping error categories to counts
        """
        error_counts = {}
        for result in results:
            category = result.error_category or ErrorCategory.UNKNOWN
            error_counts[category] = error_counts.get(category, 0) + 1

        return error_counts
