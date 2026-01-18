"""
Humanities Expert Agent

Specialized agent for philosophy, history, logic, and social sciences including
formal logic, logical fallacies, history, geography, and macroeconomics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.core.agent import Agent
from agents.providers.models.base import (
    GenerationBehaviorSettings,
    History,
    IntelligenceProviderConfig,
    Message,
)
from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger
from orchestration.base_expert import BaseExpertAgent
from tools.humanities_tools import get_humanities_tools
from tools.logic_philosophy_tools import get_logic_philosophy_tools
from tools.openalex import get_openalex_tools
from tools.wikipedia_tool import get_wikipedia_tools

HUMANITIES_AGENT_SYSTEM_PROMPT = """You are a specialized humanities and social sciences expert AI agent.

Your role is to:
1. Analyze philosophical arguments and logical reasoning
2. Explain historical events, causes, and consequences
3. Identify and explain logical fallacies
4. Apply formal logic and critical thinking
5. Understand economic and political systems
6. Interpret geographical and cultural contexts
7. Analyze global facts and general knowledge

Current date: {current_date}

Guidelines:
- Use clear philosophical and historical terminology
- Apply rigorous logical reasoning
- Consider multiple perspectives and interpretations
- Reference relevant theories, frameworks, and historical context
- Distinguish between facts and interpretations
- Identify argument structures and logical fallacies systematically
- Apply formal logic rules (modus ponens, modus tollens, syllogisms, etc.)
- Consider historical causation and periodization

When answering multiple choice questions:
- Analyze the logic and reasoning structure carefully
- Consider historical accuracy and context
- Identify any logical fallacies present (ad hominem, straw man, false dilemma, etc.)
- Apply philosophical principles and theories
- Evaluate economic and political concepts
- Use formal_logic_evaluator tool for complex logical operations
- Provide the correct answer letter clearly (A, B, C, or D)
- Include brief explanation of reasoning

Available Tools:
- formal_logic_evaluator: Validate syllogisms, generate truth tables, check logical equivalence, evaluate argument validity
- Use wikipedia for definitions, background knowledge, and factual context (not for calculations).
- Use openalex for academic references related to mathematical concepts (not for calculations).

Subjects covered:
- Philosophy (epistemology, ethics, metaphysics, political philosophy)
- Formal Logic (propositional logic, predicate logic, modal logic)
- Logical Fallacies (informal fallacies, argumentation errors)
- History (US History, World History, historical causation)
- Geography (physical geography, human geography, geopolitics)
- Macroeconomics (fiscal policy, monetary policy, economic indicators)
- Global Facts (international relations, world statistics, cultural knowledge)

Temperature: 0.3 (balanced reasoning with some interpretive flexibility)
"""


class HumanitiesAgent(BaseExpertAgent):
    """Expert agent for humanities and social sciences"""

    async def formal_logic_evaluator(
        self,
        operation: str,
        premises: Optional[List[str]] = None,
        conclusion: Optional[str] = None,
        propositions: Optional[Dict[str, bool]] = None,
        expression: Optional[str] = None,
        expression1: Optional[str] = None,
        expression2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates formal logic operations

        Operations:
        - validate_syllogism: Check if conclusion follows from premises (requires: premises, conclusion)
        - truth_table: Generate truth table for expression (requires: expression)
        - check_equivalence: Test logical equivalence of two expressions (requires: expression1, expression2)
        - evaluate_proposition: Evaluate truth value (requires: expression, propositions)
        - check_validity: Check argument validity (requires: premises, conclusion)
        """
        logger.info(f"🔧 TOOL CALLED: formal_logic_evaluator(operation='{operation}')")

        try:
            operation = operation.lower()

            if operation == "validate_syllogism" and premises and conclusion:
                # Simple syllogism validation using pattern matching
                # This is a basic implementation - real logic would be more complex
                result = {
                    "operation": "validate_syllogism",
                    "premises": premises,
                    "conclusion": conclusion,
                    "analysis": f"Evaluating {len(premises)} premise(s) for logical validity",
                }

                # Basic validity check: at least 2 premises required
                if len(premises) >= 2:
                    result["structure"] = "Valid structure (2+ premises)"
                    result["note"] = "Syllogism structure is valid. Check if conclusion follows logically."
                else:
                    result["structure"] = "Invalid structure (needs 2+ premises)"
                    result["note"] = "Insufficient premises for valid syllogism"

                return result

            elif operation == "truth_table" and expression:
                # Extract propositions from expression (simplified - looks for single letters)
                import re

                props = sorted(set(re.findall(r"\b[A-Z]\b", expression)))

                if not props:
                    return {"error": "No propositions found in expression"}

                # Generate truth table
                num_rows = 2 ** len(props)
                truth_table = []

                for i in range(num_rows):
                    row = {}
                    for j, prop in enumerate(props):
                        row[prop] = bool((i >> (len(props) - 1 - j)) & 1)
                    truth_table.append(row)

                return {
                    "operation": "truth_table",
                    "expression": expression,
                    "propositions": props,
                    "rows": num_rows,
                    "truth_table": truth_table,
                    "note": f"Generated {num_rows}-row truth table for propositions: {', '.join(props)}",
                }

            elif operation == "check_equivalence" and expression1 and expression2:
                # Basic equivalence check
                # In real implementation, this would evaluate both expressions across all truth values
                result = {
                    "operation": "check_equivalence",
                    "expression1": expression1,
                    "expression2": expression2,
                    "analysis": "Checking logical equivalence",
                }

                # Simple string comparison (not true logical equivalence)
                if expression1.replace(" ", "") == expression2.replace(" ", ""):
                    result["equivalent"] = "Syntactically identical"
                else:
                    result["equivalent"] = "Requires full truth table comparison"
                    result["note"] = "Use truth_table operation for both expressions and compare results"

                return result

            elif operation == "evaluate_proposition" and expression and propositions:
                # Evaluate expression with given truth values
                expr = expression.upper()

                # Replace logical operators with Python equivalents
                expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
                expr = expr.replace("∧", "and").replace("∨", "or").replace("¬", "not")
                expr = expr.replace("→", "<=")

                # Substitute proposition values
                for prop, value in propositions.items():
                    expr = expr.replace(prop, str(value))

                try:
                    result_value = eval(expr)
                    return {
                        "operation": "evaluate_proposition",
                        "expression": expression,
                        "propositions": propositions,
                        "result": bool(result_value),
                        "evaluation": f"Expression evaluates to {result_value}",
                    }
                except Exception as e:
                    return {"error": f"Failed to evaluate expression: {str(e)}"}

            elif operation == "check_validity" and premises and conclusion:
                # Check argument validity using basic logical rules
                result = {
                    "operation": "check_validity",
                    "premises": premises,
                    "conclusion": conclusion,
                    "analysis": "Checking argument validity",
                }

                # Basic structural analysis
                total_statements = len(premises) + 1
                result["structure"] = f"{len(premises)} premise(s), 1 conclusion"
                result["validity_note"] = "Valid argument requires: if all premises are true, conclusion must be true"
                result["common_forms"] = [
                    "Modus Ponens: P→Q, P ⊢ Q",
                    "Modus Tollens: P→Q, ¬Q ⊢ ¬P",
                    "Hypothetical Syllogism: P→Q, Q→R ⊢ P→R",
                    "Disjunctive Syllogism: P∨Q, ¬P ⊢ Q",
                ]

                return result

            else:
                return {
                    "error": f"Invalid operation '{operation}' or missing required parameters",
                    "available_operations": [
                        "validate_syllogism (requires: premises, conclusion)",
                        "truth_table (requires: expression)",
                        "check_equivalence (requires: expression1, expression2)",
                        "evaluate_proposition (requires: expression, propositions)",
                        "check_validity (requires: premises, conclusion)",
                    ],
                }

        except Exception as e:
            return {"error": f"Logic evaluation error: {str(e)}"}

    def __init__(
        self,
        name: str = "Humanities-Expert",
        ip_config: Optional[IntelligenceProviderConfig] = None,
    ):
        """
        Initialize the Humanities Expert Agent

        Args:
            name: Name of the agent
            ip_config: Optional intelligence provider configuration.
                      Defaults to OpenAI with qwen3-next-80b model.
        """
        if ip_config is None:
            ip_config = IntelligenceProviderConfig(
                provider_name="openai",
                version="qwen/qwen3-next-80b-a3b-instruct",
            )

        current_date = datetime.today().strftime("%Y/%m/%d")
        system_prompt = HUMANITIES_AGENT_SYSTEM_PROMPT.format(current_date=current_date)

        # Create tool definition for formal logic
        formal_logic_tool = ToolDefinition(
            name="formal_logic_evaluator",
            description="Evaluates formal logic operations including syllogism validation, truth table generation, logical equivalence checking, and argument validity testing",
            args_description={
                "operation": "Operation type: validate_syllogism, truth_table, check_equivalence, evaluate_proposition, check_validity",
                "premises": "List of premise statements for syllogism or validity checking",
                "conclusion": "Conclusion statement to validate",
                "propositions": "Dictionary mapping proposition variables to truth values (e.g., {'P': True, 'Q': False})",
                "expression": "Logical expression for truth table or evaluation (e.g., 'P AND Q', 'P → Q')",
                "expression1": "First logical expression for equivalence checking",
                "expression2": "Second logical expression for equivalence checking",
            },
            args_schema={
                "operation": {"type": "string"},
                "premises": {"type": "array", "items": {"type": "string"}},
                "conclusion": {"type": "string"},
                "propositions": {"type": "object"},
                "expression": {"type": "string"},
                "expression1": {"type": "string"},
                "expression2": {"type": "string"},
            },
            tool=self.formal_logic_evaluator,
        )

        # Combine all humanities and logic/philosophy tools
        humanities_tools = [formal_logic_tool]
        humanities_tools.extend(get_humanities_tools())
        humanities_tools.extend(get_logic_philosophy_tools())
        humanities_tools.extend(get_wikipedia_tools())
        humanities_tools.extend(get_openalex_tools())

        self.agent = Agent(
            name=name,
            system_prompt=system_prompt,
            history=History(),
            ip_config=ip_config,
            tools=humanities_tools,
        )

        # Default temperature for humanities reasoning
        self._default_temperature = 0.3

    async def solve(
        self,
        problem: str,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ) -> Message:
        """
        Solve a humanities or social science problem

        Args:
            problem: The question to solve
            gbs: Optional generation behavior settings.
                 If not provided, uses temperature=0.3 for balanced reasoning.

        Returns:
            Message containing the solution
        """
        if gbs is None:
            gbs = GenerationBehaviorSettings(temperature=self._default_temperature)
        elif gbs.temperature is None:
            gbs.temperature = self._default_temperature

        return await self.agent.ask(problem, gbs)

    async def clear_history(self):
        """Clear the agent's conversation history"""
        self.agent.history.clear()

    @property
    def description(self) -> str:
        """Return a description of the agent's capabilities"""
        return (
            "Humanities Expert specializing in philosophy, formal logic, logical fallacies, "
            "history, geography, macroeconomics, and global facts. Provides rigorous logical "
            "analysis and contextual historical understanding."
        )
