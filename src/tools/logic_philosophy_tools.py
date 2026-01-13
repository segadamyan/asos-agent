"""
Production Logic and Philosophy Tools

Implements logical fallacy detection, truth tables, syllogism evaluation,
and argument structure analysis with proper algorithms.
"""

import re
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Dict, List, Optional

from agents.tools.base import ToolDefinition


class FallacyType(Enum):
    """Common logical fallacies."""

    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_DILEMMA = "false_dilemma"
    SLIPPERY_SLOPE = "slippery_slope"
    CIRCULAR_REASONING = "circular_reasoning"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    HASTY_GENERALIZATION = "hasty_generalization"
    POST_HOC = "post_hoc"
    RED_HERRING = "red_herring"
    BEGGING_THE_QUESTION = "begging_the_question"
    APPEAL_TO_EMOTION = "appeal_to_emotion"
    BANDWAGON = "bandwagon"
    FALSE_CAUSE = "false_cause"
    TU_QUOQUE = "tu_quoque"


@dataclass
class FallacyPattern:
    """Pattern for detecting logical fallacies."""

    name: str
    description: str
    indicators: List[str]
    examples: List[str]


FALLACY_PATTERNS = {
    FallacyType.AD_HOMINEM: FallacyPattern(
        name="Ad Hominem",
        description="Attacking the person instead of the argument",
        indicators=[
            r"\byou(?:'re| are) (?:stupid|dumb|ignorant|biased)\b",
            r"\bof course (?:he|she|they) (?:would say|thinks?) that\b",
            r"\bcan't trust (?:him|her|them)\b",
            r"\bconsider the source\b",
            r"\bwhat (?:do|does) (?:he|she|they) know\b",
        ],
        examples=["You're too stupid to understand", "Of course he'd say that"],
    ),
    FallacyType.STRAW_MAN: FallacyPattern(
        name="Straw Man",
        description="Misrepresenting opponent's argument",
        indicators=[
            r"\bso (?:you(?:'re| are)|what you(?:'re| are)) saying (?:is|that)\b",
            r"\byou (?:want|think|believe) that\b",
            r"\baccording to (?:you|your logic)\b",
        ],
        examples=["So you're saying we should do nothing?"],
    ),
    FallacyType.FALSE_DILEMMA: FallacyPattern(
        name="False Dilemma",
        description="Presenting only two options when more exist",
        indicators=[
            r"\beither .+ or .+\b",
            r"\byou(?:'re| are) either .+ or .+\b",
            r"\bit's .+ or .+, (?:there(?:'s| is)|that(?:'s| is)) no\b",
            r"\bonly two (?:options|choices|ways)\b",
        ],
        examples=["You're either with us or against us"],
    ),
    FallacyType.SLIPPERY_SLOPE: FallacyPattern(
        name="Slippery Slope",
        description="Claiming one action leads to extreme consequences",
        indicators=[
            r"\bif .+ then .+ and then .+ and (?:eventually|ultimately|finally)\b",
            r"\bthis (?:will|would) lead to\b",
            r"\bnext thing you know\b",
            r"\bwhere (?:does|will) it (?:end|stop)\b",
            r"\bonce .+ (?:it(?:'s| is)|there(?:'s| is)) no (?:going back|stopping)\b",
        ],
        examples=["If we allow this, next thing you know everything will collapse"],
    ),
    FallacyType.CIRCULAR_REASONING: FallacyPattern(
        name="Circular Reasoning",
        description="Conclusion restates the premise",
        indicators=[
            r"\bbecause it(?:'s| is)\b",
            r"\bit(?:'s| is) .+ because (?:it(?:'s| is)|that(?:'s| is))\b",
            r"\bthe reason .+ is because\b",
        ],
        examples=["It's true because it is"],
    ),
    FallacyType.APPEAL_TO_AUTHORITY: FallacyPattern(
        name="Appeal to Authority",
        description="Citing authority instead of evidence",
        indicators=[
            r"\bexperts? (?:say|agree|believe)\b",
            r"\bscientists? (?:say|prove|show)\b",
            r"\bstudies show\b",
            r"\beveryone knows\b",
            r"\bit(?:'s| is) (?:a )?(?:well[- ]known|proven) fact\b",
        ],
        examples=["Experts say this is true, so it must be"],
    ),
    FallacyType.HASTY_GENERALIZATION: FallacyPattern(
        name="Hasty Generalization",
        description="Drawing broad conclusions from limited evidence",
        indicators=[
            r"\ball .+ (?:are|do)\b",
            r"\bevery(?:one|body) .+ (?:is|does)\b",
            r"\bnone of .+ (?:are|do)\b",
            r"\bi (?:know|met) (?:a|one) .+ (?:and|who|so)\b",
            r"\balways\b",
            r"\bnever\b",
        ],
        examples=["I met one lawyer who lied, so all lawyers are liars"],
    ),
    FallacyType.POST_HOC: FallacyPattern(
        name="Post Hoc Ergo Propter Hoc",
        description="Assuming causation from sequence",
        indicators=[
            r"\bafter .+ (?:then|so|therefore)\b",
            r"\b(?:since|because) .+ happened .+ (?:must have )?caused\b",
            r"\bright after\b",
            r"\bcaused by\b",
        ],
        examples=["It rained after I washed my car, so washing caused rain"],
    ),
    FallacyType.RED_HERRING: FallacyPattern(
        name="Red Herring",
        description="Introducing irrelevant information",
        indicators=[
            r"\bbut what about\b",
            r"\bthe (?:real|bigger) (?:issue|problem) is\b",
            r"\binstead of .+ (?:we|let(?:'s| us)) (?:talk|discuss)\b",
        ],
        examples=["But what about this other unrelated issue?"],
    ),
    FallacyType.APPEAL_TO_EMOTION: FallacyPattern(
        name="Appeal to Emotion",
        description="Using emotions instead of logic",
        indicators=[
            r"\bthink (?:of|about) the children\b",
            r"\bhow (?:would|could) you\b",
            r"\bdon't you (?:care|feel)\b",
            r"\bimagine if\b",
        ],
        examples=["Think of the children!"],
    ),
    FallacyType.BANDWAGON: FallacyPattern(
        name="Bandwagon",
        description="Appealing to popularity",
        indicators=[
            r"\beveryone (?:is doing|does|believes)\b",
            r"\b(?:most|millions of) people\b",
            r"\bpopular opinion\b",
            r"\beveryone else\b",
        ],
        examples=["Everyone believes it, so it must be true"],
    ),
}


async def identify_logical_fallacy(
    argument: str,
    fallacy_type: Optional[str] = None,
) -> str:
    """
    Identify logical fallacies in an argument using pattern matching.

    Args:
        argument: Argument text to analyze
        fallacy_type: Specific fallacy type to check for (optional)

    Returns:
        Analysis of logical fallacies
    """
    argument_lower = argument.lower()
    detected_fallacies = []

    # Check for specific fallacy type
    if fallacy_type:
        try:
            target_fallacy = FallacyType(fallacy_type.lower())
            pattern = FALLACY_PATTERNS.get(target_fallacy)

            if pattern:
                for indicator in pattern.indicators:
                    if re.search(indicator, argument_lower, re.IGNORECASE):
                        return (
                            f"✓ Potential {pattern.name} detected\n\n"
                            f"Description: {pattern.description}\n\n"
                            f"Pattern matched: {indicator}\n\n"
                            f"Note: This is a heuristic analysis. Manual review recommended."
                        )

                return (
                    f"✗ No clear {pattern.name} pattern detected in the argument.\n\n"
                    f"Description: {pattern.description}\n\n"
                    f"Example indicators: {', '.join(pattern.examples[:2])}"
                )
        except ValueError:
            return (
                f"Unknown fallacy type: {fallacy_type}\n\nAvailable types: {', '.join([f.value for f in FallacyType])}"
            )

    # Check all fallacy types
    for fallacy, pattern in FALLACY_PATTERNS.items():
        matched_indicators = []
        for indicator in pattern.indicators:
            if re.search(indicator, argument_lower, re.IGNORECASE):
                matched_indicators.append(indicator)

        if matched_indicators:
            detected_fallacies.append(
                {
                    "name": pattern.name,
                    "description": pattern.description,
                    "matched": matched_indicators[:2],  # Limit to 2 patterns
                }
            )

    # Format results
    if detected_fallacies:
        result = f"Logical Fallacy Analysis\n{'=' * 50}\n\n"
        result += f"Argument: {argument}\n\n"
        result += f"Detected Fallacies ({len(detected_fallacies)}):\n\n"

        for i, fallacy in enumerate(detected_fallacies, 1):
            result += f"{i}. {fallacy['name']}\n"
            result += f"   {fallacy['description']}\n"
            result += f"   Matched patterns: {len(fallacy['matched'])}\n\n"

        result += (
            "\nNote: This is automated pattern detection. Context and intent matter for true fallacy identification."
        )
        return result
    else:
        return (
            f"Logical Fallacy Analysis\n{'=' * 50}\n\n"
            f"Argument: {argument}\n\n"
            f"✓ No obvious fallacy patterns detected.\n\n"
            f"The argument appears logically sound based on pattern analysis, "
            f"though deeper semantic analysis may reveal other issues."
        )


async def construct_truth_table(
    variables: List[str],
    expression: str,
) -> str:
    """
    Construct a truth table for a logical expression.

    Args:
        variables: List of variable names (e.g., ["P", "Q"])
        expression: Logical expression using &(AND), |(OR), ~(NOT), ^(XOR), ->(IMPLIES)

    Returns:
        Formatted truth table
    """
    if not variables:
        return "Error: At least one variable required"

    if len(variables) > 8:
        return f"Error: Too many variables ({len(variables)}). Maximum is 8 (256 rows)."

    # Normalize expression
    expr = expression.strip()

    # Replace logical operators with Python operators
    expr_eval = expr
    expr_eval = expr_eval.replace("→", ">>")
    expr_eval = expr_eval.replace("->", ">>")
    expr_eval = expr_eval.replace("∧", "&")
    expr_eval = expr_eval.replace("∨", "|")
    expr_eval = expr_eval.replace("¬", "~")
    expr_eval = expr_eval.replace("⊕", "^")

    # Generate all combinations
    num_vars = len(variables)
    combinations = list(product([False, True], repeat=num_vars))

    # Build table
    header = " | ".join(variables) + " | " + expression
    separator = "-" * len(header)

    rows = []
    rows.append(header)
    rows.append(separator)

    for combo in combinations:
        # Create variable mapping
        var_dict = dict(zip(variables, combo))

        # Evaluate expression
        try:
            # Replace variables in expression
            eval_expr = expr_eval
            for var, val in var_dict.items():
                eval_expr = re.sub(r"\b" + var + r"\b", str(val), eval_expr)

            # Handle implication (A >> B = ~A | B)
            while ">>" in eval_expr:
                eval_expr = re.sub(r"\(([^)]+)\)\s*>>\s*\(([^)]+)\)", r"(not (\1) or (\2))", eval_expr)
                eval_expr = re.sub(r"(\w+)\s*>>\s*(\w+)", r"(not \1 or \2)", eval_expr)

            result = eval(eval_expr)

            # Format row
            row_values = [str(int(v)) for v in combo]
            row_values.append(str(int(result)))

            # Calculate spacing
            col_widths = [len(v) for v in variables] + [len(expression)]
            row_str = " | ".join(val.center(width) for val, width in zip(row_values, col_widths))
            rows.append(row_str)

        except Exception as e:
            return (
                f"Error evaluating expression: {e}\n\n"
                f"Supported operators:\n"
                f"  & or ∧ (AND)\n"
                f"  | or ∨ (OR)\n"
                f"  ~ or ¬ (NOT)\n"
                f"  ^ or ⊕ (XOR)\n"
                f"  -> or → (IMPLIES)\n\n"
                f"Example: (P & Q) | ~R"
            )

    # Summary
    true_count = sum(1 for row in rows[2:] if row.split("|")[-1].strip() == "1")
    total_rows = len(combinations)

    result = "Truth Table\n" + "=" * 50 + "\n\n"
    result += "\n".join(rows)
    result += f"\n\nSummary: {true_count}/{total_rows} rows are TRUE"

    # Check for tautology or contradiction
    if true_count == total_rows:
        result += "\n\n✓ This is a TAUTOLOGY (always true)"
    elif true_count == 0:
        result += "\n\n✗ This is a CONTRADICTION (always false)"

    return result


async def evaluate_syllogism(
    premise1: str,
    premise2: str,
    conclusion: str,
) -> str:
    """
    Evaluate a categorical syllogism for validity.

    Args:
        premise1: First premise (e.g., "All A are B")
        premise2: Second premise (e.g., "All B are C")
        conclusion: Conclusion (e.g., "All A are C")

    Returns:
        Validity analysis of the syllogism
    """

    def parse_categorical_statement(stmt: str) -> Optional[Dict]:
        """Parse categorical statement into type and terms."""
        stmt = stmt.strip().lower()

        patterns = [
            (r"all\s+(\w+)\s+(?:are|is)\s+(\w+)", "A"),  # Universal Affirmative
            (r"no\s+(\w+)\s+(?:are|is)\s+(\w+)", "E"),  # Universal Negative
            (r"some\s+(\w+)\s+(?:are|is)\s+(\w+)", "I"),  # Particular Affirmative
            (r"some\s+(\w+)\s+(?:are|is)\s+not\s+(\w+)", "O"),  # Particular Negative
        ]

        for pattern, stmt_type in patterns:
            match = re.search(pattern, stmt)
            if match:
                return {"type": stmt_type, "subject": match.group(1), "predicate": match.group(2)}
        return None

    # Parse all statements
    p1 = parse_categorical_statement(premise1)
    p2 = parse_categorical_statement(premise2)
    conc = parse_categorical_statement(conclusion)

    if not all([p1, p2, conc]):
        return (
            "Error: Could not parse syllogism.\n\n"
            "Statements must be in one of these forms:\n"
            "  - All X are Y (Universal Affirmative - A)\n"
            "  - No X are Y (Universal Negative - E)\n"
            "  - Some X are Y (Particular Affirmative - I)\n"
            "  - Some X are not Y (Particular Negative - O)\n\n"
            "Example:\n"
            "  Premise 1: All humans are mortal\n"
            "  Premise 2: All Greeks are humans\n"
            "  Conclusion: All Greeks are mortal"
        )

    # Identify terms
    all_terms = set()
    all_terms.update([p1["subject"], p1["predicate"]])
    all_terms.update([p2["subject"], p2["predicate"]])
    all_terms.update([conc["subject"], conc["predicate"]])

    # Find middle term (appears in both premises but not conclusion)
    premise_terms = set([p1["subject"], p1["predicate"], p2["subject"], p2["predicate"]])
    conclusion_terms = set([conc["subject"], conc["predicate"]])
    middle_terms = premise_terms - conclusion_terms

    # Check basic validity conditions
    errors = []
    warnings = []

    # Rule 1: Must have exactly 3 terms
    if len(all_terms) != 3:
        errors.append(f"Four-term fallacy: Found {len(all_terms)} terms, need exactly 3")

    # Rule 2: Middle term must be distributed at least once
    if middle_terms:
        middle_term = list(middle_terms)[0]
        middle_distributed = False

        # Check if middle term is distributed in premises
        if p1["subject"] == middle_term and p1["type"] in ["A", "E"]:
            middle_distributed = True
        if p1["predicate"] == middle_term and p1["type"] == "E":
            middle_distributed = True
        if p2["subject"] == middle_term and p2["type"] in ["A", "E"]:
            middle_distributed = True
        if p2["predicate"] == middle_term and p2["type"] == "E":
            middle_distributed = True

        if not middle_distributed:
            errors.append("Undistributed middle: Middle term not distributed in premises")

    # Rule 3: If term is distributed in conclusion, must be distributed in premise
    if conc["type"] in ["A", "E"]:  # Subject distributed in conclusion
        subject_dist_in_premises = False
        for p in [p1, p2]:
            if p["subject"] == conc["subject"] and p["type"] in ["A", "E"]:
                subject_dist_in_premises = True
            if p["predicate"] == conc["subject"] and p["type"] == "E":
                subject_dist_in_premises = True

        if not subject_dist_in_premises:
            errors.append("Illicit major/minor: Term distributed in conclusion but not in premise")

    # Rule 4: Two negative premises yield no conclusion
    if p1["type"] in ["E", "O"] and p2["type"] in ["E", "O"]:
        errors.append("Two negative premises: Cannot draw valid conclusion")

    # Rule 5: Negative premise requires negative conclusion
    if (p1["type"] in ["E", "O"] or p2["type"] in ["E", "O"]) and conc["type"] not in ["E", "O"]:
        errors.append("Negative premise but affirmative conclusion")

    # Format output
    result = "Syllogism Evaluation\n" + "=" * 50 + "\n\n"
    result += f"Premise 1: {premise1}\n"
    result += f"  Type: {p1['type']} (Subject: {p1['subject']}, Predicate: {p1['predicate']})\n\n"
    result += f"Premise 2: {premise2}\n"
    result += f"  Type: {p2['type']} (Subject: {p2['subject']}, Predicate: {p2['predicate']})\n\n"
    result += f"Conclusion: {conclusion}\n"
    result += f"  Type: {conc['type']} (Subject: {conc['subject']}, Predicate: {conc['predicate']})\n\n"

    if middle_terms:
        result += f"Middle term: {list(middle_terms)[0]}\n\n"

    result += "Validity Analysis:\n"
    result += "-" * 50 + "\n"

    if errors:
        result += "\n✗ INVALID\n\n"
        result += "Errors found:\n"
        for i, error in enumerate(errors, 1):
            result += f"  {i}. {error}\n"
    else:
        result += "\n✓ VALID\n\n"
        result += "The syllogism follows the rules of categorical logic.\n"

    if warnings:
        result += "\nWarnings:\n"
        for warning in warnings:
            result += f"  - {warning}\n"

    return result


async def analyze_argument_structure(
    argument: str,
) -> str:
    """
    Analyze the structure of an argument (premises and conclusion).

    Args:
        argument: Argument text

    Returns:
        Identified premises, conclusion, and structural analysis
    """

    # Indicators for conclusions
    conclusion_indicators = [
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bso\b",
        r"\bconsequently\b",
        r"\bin conclusion\b",
        r"\bit follows that\b",
        r"\bwe can conclude\b",
        r"\bthis (?:shows|proves|means) that\b",
    ]

    # Indicators for premises
    premise_indicators = [
        r"\bbecause\b",
        r"\bsince\b",
        r"\bgiven that\b",
        r"\bas\b",
        r"\bfor\b",
        r"\bthe reason is\b",
        r"\bfirstly\b",
        r"\bsecondly\b",
    ]

    sentences = re.split(r"[.!?]+", argument)
    sentences = [s.strip() for s in sentences if s.strip()]

    premises = []
    conclusions = []
    neutral = []

    for sent in sentences:
        sent_lower = sent.lower()

        # Check for conclusion indicators
        is_conclusion = any(re.search(ind, sent_lower) for ind in conclusion_indicators)

        # Check for premise indicators
        is_premise = any(re.search(ind, sent_lower) for ind in premise_indicators)

        if is_conclusion:
            conclusions.append(sent)
        elif is_premise:
            premises.append(sent)
        else:
            neutral.append(sent)

    # If no explicit indicators, make educated guess
    if not conclusions and sentences:
        # Often conclusion is last sentence
        conclusions.append(sentences[-1])
        premises = sentences[:-1]
        neutral = []

    # Format output
    result = "Argument Structure Analysis\n" + "=" * 50 + "\n\n"
    result += f"Original Argument:\n{argument}\n\n"
    result += "=" * 50 + "\n\n"

    if premises:
        result += f"Premises ({len(premises)}):\n"
        for i, premise in enumerate(premises, 1):
            result += f"  P{i}: {premise}\n"
        result += "\n"
    else:
        result += "Premises: None explicitly identified\n\n"

    if conclusions:
        result += f"Conclusion(s) ({len(conclusions)}):\n"
        for i, conclusion in enumerate(conclusions, 1):
            result += f"  C{i}: {conclusion}\n"
        result += "\n"
    else:
        result += "Conclusion: None explicitly identified\n\n"

    if neutral:
        result += f"Other Statements ({len(neutral)}):\n"
        for i, stmt in enumerate(neutral, 1):
            result += f"  {i}. {stmt}\n"
        result += "\n"

    # Structural assessment
    result += "=" * 50 + "\n"
    result += "Structural Assessment:\n\n"

    if premises and conclusions:
        result += "✓ Argument has identifiable structure\n"
        result += f"  - {len(premises)} premise(s) supporting {len(conclusions)} conclusion(s)\n"
    else:
        result += "⚠ Limited explicit structure\n"
        result += "  - Consider adding indicator words (therefore, because, since)\n"

    # Check for common patterns
    if len(premises) >= 2 and len(conclusions) == 1:
        result += "  - Pattern: Multiple premises → Single conclusion (Deductive)\n"
    elif len(premises) == 1 and len(conclusions) == 1:
        result += "  - Pattern: Single premise → Single conclusion (Simple)\n"

    result += "\nNote: Analysis based on linguistic indicators. Implicit premises may exist."

    return result


async def evaluate_logical_statement(
    statement: str,
    truth_values: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Evaluate a logical statement (basic implementation).

    Args:
        statement: Logical statement (e.g., "P AND Q", "NOT P", "P OR Q")
        truth_values: Dictionary of truth values for variables (e.g., {"P": True, "Q": False})

    Returns:
        Truth value of the statement
    """
    # This is a simplified implementation - for complex logic, use a proper parser
    if truth_values is None:
        truth_values = {}

    # Simple evaluation for basic logical operators
    try:
        # Replace variables with their truth values
        eval_statement = statement.upper()
        for var, val in truth_values.items():
            eval_statement = eval_statement.replace(var.upper(), str(val))

        # Replace logical operators with Python operators
        eval_statement = eval_statement.replace("AND", "and")
        eval_statement = eval_statement.replace("OR", "or")
        eval_statement = eval_statement.replace("NOT", "not")
        eval_statement = eval_statement.replace("XOR", "^")

        # Evaluate
        result = eval(eval_statement)

        return (
            f"Logical Statement Evaluation\n{'=' * 50}\n\n"
            f"Statement: {statement}\n"
            f"Truth Values: {truth_values}\n\n"
            f"Result: {result}\n\n"
            f"Note: This is a basic implementation. For complex logic, use construct_truth_table."
        )
    except Exception as e:
        return (
            f"Error evaluating logical statement: {e}\n\n"
            f"Supported operators: AND, OR, NOT, XOR\n"
            f"Example: 'P AND Q' with truth_values={{'P': True, 'Q': False}}"
        )


def get_logic_philosophy_tools() -> List[ToolDefinition]:
    """Get all logic and philosophy tool definitions."""
    return [
        ToolDefinition(
            name="evaluate_logical_statement",
            description="Evaluate a logical statement with given truth values (basic implementation).",
            args_description={
                "statement": "Logical statement (e.g., 'P AND Q', 'NOT P', 'P OR Q')",
                "truth_values": "Dictionary of truth values for variables (e.g., {'P': True, 'Q': False})",
            },
            args_schema={
                "statement": {"type": "string"},
                "truth_values": {"type": "object"},
            },
            tool=evaluate_logical_statement,
        ),
        ToolDefinition(
            name="identify_logical_fallacy",
            description="Identify logical fallacies in an argument using pattern matching. Detects common fallacies like ad hominem, straw man, false dilemma, etc.",
            args_description={
                "argument": "Argument text to analyze",
                "fallacy_type": "Specific fallacy type to check for (optional). Available: ad_hominem, straw_man, false_dilemma, slippery_slope, circular_reasoning, appeal_to_authority, hasty_generalization, post_hoc, red_herring, begging_the_question, appeal_to_emotion, bandwagon",
            },
            args_schema={
                "argument": {"type": "string"},
                "fallacy_type": {"type": "string"},
            },
            tool=identify_logical_fallacy,
        ),
        ToolDefinition(
            name="construct_truth_table",
            description="Construct a truth table for a logical expression. Supports AND (&), OR (|), NOT (~), XOR (^), and IMPLIES (->).",
            args_description={
                "variables": "List of variable names (e.g., ['P', 'Q'])",
                "expression": "Logical expression using &(AND), |(OR), ~(NOT), ^(XOR), ->(IMPLIES)",
            },
            args_schema={
                "variables": {"type": "array", "items": {"type": "string"}},
                "expression": {"type": "string"},
            },
            tool=construct_truth_table,
        ),
        ToolDefinition(
            name="evaluate_syllogism",
            description="Evaluate a categorical syllogism for validity. Analyzes premises and conclusion according to rules of categorical logic.",
            args_description={
                "premise1": "First premise (e.g., 'All A are B')",
                "premise2": "Second premise (e.g., 'All B are C')",
                "conclusion": "Conclusion (e.g., 'All A are C')",
            },
            args_schema={
                "premise1": {"type": "string"},
                "premise2": {"type": "string"},
                "conclusion": {"type": "string"},
            },
            tool=evaluate_syllogism,
        ),
        ToolDefinition(
            name="analyze_argument_structure",
            description="Analyze the structure of an argument, identifying premises and conclusion based on linguistic indicators.",
            args_description={
                "argument": "Argument text",
            },
            args_schema={
                "argument": {"type": "string"},
            },
            tool=analyze_argument_structure,
        ),
    ]
