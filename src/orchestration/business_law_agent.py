"""
Business & Law Expert Agent

Specialized agent for business, legal, and professional subjects including
business ethics, professional accounting, corporate law, and public relations.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.core.simple import SimpleAgent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig, Message
from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger
from orchestration.base_expert import BaseExpertAgent

BUSINESS_LAW_AGENT_SYSTEM_PROMPT = """You are a specialized business, law, and professional ethics expert AI agent.

Your role is to:
1. Answer questions about business ethics, professional accounting, and corporate law
2. Provide accurate legal reasoning and professional standards
3. Explain accounting principles and financial regulations
4. Handle questions about professional conduct and ethics
5. Apply logical reasoning to legal and business scenarios
6. Analyze economic regulations and policies

Current date: {current_date}

Guidelines:
- Use proper legal and business terminology
- Cite relevant principles when applicable
- Consider professional ethics codes (AICPA, ABA, etc.)
- Explain regulatory frameworks clearly (SEC, GAAP, etc.)
- Distinguish between legal requirements and ethical best practices
- Apply case law and statutory interpretation when relevant
- Consider corporate governance principles

When answering multiple choice questions:
- Analyze each option carefully using legal/business reasoning
- Consider relevant statutes, regulations, and professional standards
- Use available tools for complex calculations (financial, statistical)
- Apply legal frameworks like IRAC, contract elements, tort analysis when relevant
- Identify the most legally/ethically sound answer
- Provide the correct answer letter clearly (A, B, C, or D)
- Include brief reasoning based on legal/business principles

Available Tools:
- financial_calculator: NPV, IRR, financial ratios, present/future value calculations
- statistical_calculator: Correlation, regression, hypothesis testing, econometric analysis

Subjects covered:
- Business Ethics (corporate responsibility, stakeholder theory, ethical decision-making)
- Professional Accounting (GAAP, financial reporting, auditing standards)
- Professional Law (contracts, torts, corporate law, professional liability)
- Econometrics (economic modeling, statistical analysis)
- Public Relations (corporate communications, crisis management)

Temperature: 0.2 (precise, rule-based reasoning required)
"""


class BusinessLawAgent(BaseExpertAgent):
    """Expert agent for business, law, and professional subjects"""

    async def financial_calculator(
        self,
        operation: str,
        principal: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
        cash_flows: Optional[List[float]] = None,
        current_assets: Optional[float] = None,
        current_liabilities: Optional[float] = None,
        total_debt: Optional[float] = None,
        total_equity: Optional[float] = None,
        net_income: Optional[float] = None,
        revenue: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Performs financial and accounting calculations

        Operations:
        - present_value: Calculate PV (requires: cash_flows, rate)
        - future_value: Calculate FV (requires: principal, rate, time)
        - npv: Net Present Value (requires: cash_flows, rate)
        - irr: Internal Rate of Return (requires: cash_flows)
        - current_ratio: Current Assets / Current Liabilities
        - debt_to_equity: Total Debt / Total Equity
        - profit_margin: Net Income / Revenue
        - roe: Return on Equity = Net Income / Total Equity
        - roi: Return on Investment
        """
        logger.info(f"🔧 TOOL CALLED: financial_calculator(operation='{operation}')")
        try:
            operation = operation.lower()

            if operation == "present_value" and cash_flows and rate is not None:
                pv = sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                return {"result": round(pv, 2), "formula": "PV = Σ(CF_t / (1+r)^t)"}

            elif operation == "future_value" and principal and rate is not None and time:
                fv = principal * ((1 + rate) ** time)
                return {"result": round(fv, 2), "formula": "FV = PV * (1+r)^t"}

            elif operation == "npv" and cash_flows and rate is not None:
                npv = sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                initial_investment = cash_flows[0] if cash_flows else 0
                npv_result = npv + initial_investment
                return {"result": round(npv_result, 2), "formula": "NPV = Σ(CF_t / (1+r)^t) - Initial Investment"}

            elif operation == "irr" and cash_flows:
                # Simple IRR approximation using Newton-Raphson
                def npv_at_rate(r):
                    return sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cash_flows))

                rate_guess = 0.1
                for _ in range(100):
                    npv_val = npv_at_rate(rate_guess)
                    if abs(npv_val) < 0.01:
                        break
                    # Derivative approximation
                    npv_plus = npv_at_rate(rate_guess + 0.001)
                    derivative = (npv_plus - npv_val) / 0.001
                    if derivative == 0:
                        break
                    rate_guess = rate_guess - npv_val / derivative

                return {"result": round(rate_guess * 100, 2), "unit": "%", "formula": "IRR: rate where NPV = 0"}

            elif operation == "current_ratio" and current_assets and current_liabilities:
                ratio = current_assets / current_liabilities if current_liabilities != 0 else "Undefined"
                return {
                    "result": round(ratio, 2) if isinstance(ratio, float) else ratio,
                    "formula": "Current Ratio = Current Assets / Current Liabilities",
                }

            elif operation == "debt_to_equity" and total_debt is not None and total_equity:
                ratio = total_debt / total_equity if total_equity != 0 else "Undefined"
                return {
                    "result": round(ratio, 2) if isinstance(ratio, float) else ratio,
                    "formula": "Debt-to-Equity = Total Debt / Total Equity",
                }

            elif operation == "profit_margin" and net_income is not None and revenue:
                margin = (net_income / revenue) * 100 if revenue != 0 else "Undefined"
                return {
                    "result": round(margin, 2) if isinstance(margin, float) else margin,
                    "unit": "%",
                    "formula": "Profit Margin = (Net Income / Revenue) * 100",
                }

            elif operation == "roe" and net_income is not None and total_equity:
                roe = (net_income / total_equity) * 100 if total_equity != 0 else "Undefined"
                return {
                    "result": round(roe, 2) if isinstance(roe, float) else roe,
                    "unit": "%",
                    "formula": "ROE = (Net Income / Total Equity) * 100",
                }

            else:
                return {"error": f"Invalid operation or missing required parameters for {operation}"}

        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}

    async def statistical_calculator(
        self,
        operation: str,
        data: Optional[List[float]] = None,
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
        sample1: Optional[List[float]] = None,
        sample2: Optional[List[float]] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Performs statistical and econometric calculations

        Operations:
        - mean: Calculate average
        - median: Calculate median
        - std_dev: Standard deviation
        - variance: Variance
        - correlation: Pearson correlation (requires: x_data, y_data)
        - regression: Simple linear regression (requires: x_data, y_data)
        - t_test: Two-sample t-test (requires: sample1, sample2)
        - z_score: Calculate z-scores
        """

        try:
            operation = operation.lower()

            if operation == "mean" and data:
                mean = sum(data) / len(data)
                return {"result": round(mean, 4), "formula": "Mean = Σx / n"}

            elif operation == "median" and data:
                sorted_data = sorted(data)
                n = len(sorted_data)
                median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
                return {"result": round(median, 4), "formula": "Median = middle value"}

            elif operation == "std_dev" and data:
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                std_dev = math.sqrt(variance)
                return {"result": round(std_dev, 4), "formula": "σ = √(Σ(x-μ)² / n)"}

            elif operation == "variance" and data:
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                return {"result": round(variance, 4), "formula": "Var = Σ(x-μ)² / n"}

            elif operation == "correlation" and x_data and y_data and len(x_data) == len(y_data):
                n = len(x_data)
                mean_x = sum(x_data) / n
                mean_y = sum(y_data) / n

                numerator = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
                denominator = math.sqrt(sum((x - mean_x) ** 2 for x in x_data) * sum((y - mean_y) ** 2 for y in y_data))

                correlation = numerator / denominator if denominator != 0 else 0
                return {"result": round(correlation, 4), "formula": "r = Σ((x-x̄)(y-ȳ)) / √(Σ(x-x̄)²Σ(y-ȳ)²)"}

            elif operation == "regression" and x_data and y_data and len(x_data) == len(y_data):
                n = len(x_data)
                mean_x = sum(x_data) / n
                mean_y = sum(y_data) / n

                # Calculate slope (beta)
                numerator = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
                denominator = sum((x - mean_x) ** 2 for x in x_data)
                slope = numerator / denominator if denominator != 0 else 0

                # Calculate intercept (alpha)
                intercept = mean_y - slope * mean_x

                # Calculate R-squared
                y_pred = [intercept + slope * x for x in x_data]
                ss_res = sum((y_data[i] - y_pred[i]) ** 2 for i in range(n))
                ss_tot = sum((y - mean_y) ** 2 for y in y_data)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                return {
                    "slope": round(slope, 4),
                    "intercept": round(intercept, 4),
                    "r_squared": round(r_squared, 4),
                    "equation": f"y = {round(slope, 4)}x + {round(intercept, 4)}",
                    "formula": "y = βx + α",
                }

            elif operation == "t_test" and sample1 and sample2:
                n1, n2 = len(sample1), len(sample2)
                mean1 = sum(sample1) / n1
                mean2 = sum(sample2) / n2

                var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1) if n1 > 1 else 0
                var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1) if n2 > 1 else 0

                pooled_std = math.sqrt((var1 / n1) + (var2 / n2))
                t_statistic = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

                return {
                    "t_statistic": round(t_statistic, 4),
                    "mean1": round(mean1, 4),
                    "mean2": round(mean2, 4),
                    "formula": "t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)",
                }

            elif operation == "z_score" and data:
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                std_dev = math.sqrt(variance)

                z_scores = [(x - mean) / std_dev if std_dev != 0 else 0 for x in data]
                return {
                    "z_scores": [round(z, 4) for z in z_scores],
                    "mean": round(mean, 4),
                    "std_dev": round(std_dev, 4),
                    "formula": "z = (x - μ) / σ",
                }

            else:
                return {"error": f"Invalid operation or missing parameters for {operation}"}

        except Exception as e:
            return {"error": f"Statistical calculation error: {str(e)}"}

    def __init__(
        self,
        name: str = "BusinessLaw-Expert",
        ip_config: Optional[IntelligenceProviderConfig] = None,
    ):
        """
        Initialize the Business & Law Expert Agent

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
        system_prompt = BUSINESS_LAW_AGENT_SYSTEM_PROMPT.format(current_date=current_date)

        # Create tool definitions that reference instance methods
        financial_calculator_tool = ToolDefinition(
            name="financial_calculator",
            description="Performs financial and accounting calculations including NPV, IRR, financial ratios, present/future value, and profitability metrics",
            args_description={
                "operation": "Operation type: present_value, future_value, npv, irr, current_ratio, debt_to_equity, profit_margin, roe, roi",
                "principal": "Principal amount for FV calculation",
                "rate": "Interest/discount rate (as decimal, e.g., 0.10 for 10%)",
                "time": "Time period",
                "cash_flows": "List of cash flows for NPV/IRR calculations",
                "current_assets": "Current assets for ratio analysis",
                "current_liabilities": "Current liabilities for ratio analysis",
                "total_debt": "Total debt for leverage ratios",
                "total_equity": "Total equity for leverage and return ratios",
                "net_income": "Net income for profitability ratios",
                "revenue": "Revenue for margin calculations",
            },
            args_schema={
                "operation": {"type": "string"},
                "principal": {"type": "number"},
                "rate": {"type": "number"},
                "time": {"type": "number"},
                "cash_flows": {"type": "array", "items": {"type": "number"}},
                "current_assets": {"type": "number"},
                "current_liabilities": {"type": "number"},
                "total_debt": {"type": "number"},
                "total_equity": {"type": "number"},
                "net_income": {"type": "number"},
                "revenue": {"type": "number"},
            },
            tool=self.financial_calculator,
        )

        statistical_calculator_tool = ToolDefinition(
            name="statistical_calculator",
            description="Performs statistical and econometric calculations including mean, median, standard deviation, correlation, regression, t-tests, and z-scores",
            args_description={
                "operation": "Operation type: mean, median, std_dev, variance, correlation, regression, t_test, z_score",
                "data": "Single dataset for univariate statistics",
                "x_data": "X variable data for bivariate analysis",
                "y_data": "Y variable data for bivariate analysis",
                "sample1": "First sample for t-test",
                "sample2": "Second sample for t-test",
                "confidence_level": "Confidence level (default: 0.95)",
            },
            args_schema={
                "operation": {"type": "string"},
                "data": {"type": "array", "items": {"type": "number"}},
                "x_data": {"type": "array", "items": {"type": "number"}},
                "y_data": {"type": "array", "items": {"type": "number"}},
                "sample1": {"type": "array", "items": {"type": "number"}},
                "sample2": {"type": "array", "items": {"type": "number"}},
                "confidence_level": {"type": "number"},
            },
            tool=self.statistical_calculator,
        )

        self.agent = SimpleAgent(
            name=name,
            system_prompt=system_prompt,
            history=History(),
            ip_config=ip_config,
            tools=[financial_calculator_tool, statistical_calculator_tool],
        )

        # Override default temperature for precise legal/business reasoning
        self._default_temperature = 0.2

    async def solve(
        self,
        problem: str,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ) -> Message:
        """
        Solve a business or law problem

        Args:
            problem: The business/law question to solve
            gbs: Optional generation behavior settings.
                 If not provided, uses temperature=0.2 for precise reasoning.

        Returns:
            Message containing the solution
        """
        if gbs is None:
            gbs = GenerationBehaviorSettings(temperature=self._default_temperature)
        elif gbs.temperature is None:
            gbs.temperature = self._default_temperature

        return await self.agent.answer_to(problem, gbs)

    async def clear_history(self):
        """Clear the agent's conversation history"""
        self.agent.history.clear()

    @property
    def description(self) -> str:
        """Return a description of the agent's capabilities"""
        return (
            "Business & Law Expert specializing in business ethics, professional accounting, "
            "corporate law, econometrics, and public relations. Provides precise legal and "
            "business reasoning based on professional standards and regulations."
        )
