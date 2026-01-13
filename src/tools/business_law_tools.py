"""
Business and Law Tools

"""

from typing import List, Optional

from agents.tools.base import ToolDefinition


async def calculate_financial_ratio(
    ratio_type: str,
    current_assets: Optional[float] = None,
    current_liabilities: Optional[float] = None,
    total_debt: Optional[float] = None,
    total_equity: Optional[float] = None,
    net_income: Optional[float] = None,
    revenue: Optional[float] = None,
    total_assets: Optional[float] = None,
) -> str:
    """
    Calculate financial ratios.

    Args:
        ratio_type: Type of ratio: current_ratio, debt_to_equity, profit_margin, roe, roa, quick_ratio
        current_assets: Current assets
        current_liabilities: Current liabilities
        total_debt: Total debt
        total_equity: Total equity
        net_income: Net income
        revenue: Revenue
        total_assets: Total assets

    Returns:
        Calculated ratio
    """
    try:
        ratio_type = ratio_type.lower()

        if ratio_type == "current_ratio":
            if current_assets is None or current_liabilities is None:
                return "Error: current_assets and current_liabilities required"
            ratio = current_assets / current_liabilities if current_liabilities != 0 else float("inf")
            return f"Current Ratio: {ratio:.4f} (Current Assets / Current Liabilities)"

        elif ratio_type == "quick_ratio":
            # Simplified - assumes cash + marketable securities = current_assets - inventory
            if current_assets is None or current_liabilities is None:
                return "Error: current_assets and current_liabilities required"
            ratio = current_assets / current_liabilities if current_liabilities != 0 else float("inf")
            return f"Quick Ratio: {ratio:.4f} (approximate, assumes all current assets are liquid)"

        elif ratio_type == "debt_to_equity":
            if total_debt is None or total_equity is None:
                return "Error: total_debt and total_equity required"
            ratio = total_debt / total_equity if total_equity != 0 else float("inf")
            return f"Debt-to-Equity Ratio: {ratio:.4f} (Total Debt / Total Equity)"

        elif ratio_type == "profit_margin":
            if net_income is None or revenue is None:
                return "Error: net_income and revenue required"
            margin = (net_income / revenue) * 100 if revenue != 0 else float("inf")
            return f"Profit Margin: {margin:.2f}% (Net Income / Revenue)"

        elif ratio_type == "roe":
            if net_income is None or total_equity is None:
                return "Error: net_income and total_equity required"
            roe = (net_income / total_equity) * 100 if total_equity != 0 else float("inf")
            return f"Return on Equity (ROE): {roe:.2f}% (Net Income / Total Equity)"

        elif ratio_type == "roa":
            if net_income is None or total_assets is None:
                return "Error: net_income and total_assets required"
            roa = (net_income / total_assets) * 100 if total_assets != 0 else float("inf")
            return f"Return on Assets (ROA): {roa:.2f}% (Net Income / Total Assets)"

        else:
            return f"Unknown ratio type: {ratio_type}. Available: current_ratio, quick_ratio, debt_to_equity, profit_margin, roe, roa"

    except Exception as e:
        return f"Error calculating financial ratio: {str(e)}"


async def calculate_npv(
    cash_flows: List[float],
    discount_rate: float,
    initial_investment: Optional[float] = None,
) -> str:
    """
    Calculate Net Present Value (NPV).

    Args:
        cash_flows: List of cash flows (first can be negative for initial investment)
        discount_rate: Discount rate as decimal (e.g., 0.10 for 10%)
        initial_investment: Initial investment (optional, can be included as first cash flow)

    Returns:
        NPV value
    """
    try:
        if initial_investment is not None:
            npv = -initial_investment
            for i, cf in enumerate(cash_flows):
                npv += cf / ((1 + discount_rate) ** (i + 1))
        else:
            npv = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows))

        return f"NPV: {npv:.2f} (at {discount_rate * 100}% discount rate)"
    except Exception as e:
        return f"Error calculating NPV: {str(e)}"


async def calculate_irr(cash_flows: List[float]) -> str:
    """
    Calculate Internal Rate of Return (IRR) using approximation.

    Args:
        cash_flows: List of cash flows (first should be negative for initial investment)

    Returns:
        IRR as percentage
    """
    try:

        def npv_at_rate(r):
            return sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cash_flows))

        rate_guess = 0.1
        for _ in range(100):
            npv_val = npv_at_rate(rate_guess)
            if abs(npv_val) < 0.0001:
                break
            # Derivative approximation
            npv_plus = npv_at_rate(rate_guess + 0.0001)
            derivative = (npv_plus - npv_val) / 0.0001
            if abs(derivative) < 0.0001:
                break
            rate_guess = rate_guess - npv_val / derivative

        return f"IRR: {rate_guess * 100:.2f}%"
    except Exception as e:
        return f"Error calculating IRR: {str(e)}"


async def calculate_present_value(
    future_value: float,
    rate: float,
    periods: int,
) -> str:
    """
    Calculate present value using PV = FV / (1 + r)^n.

    Args:
        future_value: Future value
        rate: Interest rate per period as decimal
        periods: Number of periods

    Returns:
        Present value
    """
    try:
        pv = future_value / ((1 + rate) ** periods)
        return f"Present Value: {pv:.2f} (PV = FV / (1 + r)^n)"
    except Exception as e:
        return f"Error calculating present value: {str(e)}"


async def calculate_future_value(
    present_value: float,
    rate: float,
    periods: int,
) -> str:
    """
    Calculate future value using FV = PV * (1 + r)^n.

    Args:
        present_value: Present value
        rate: Interest rate per period as decimal
        periods: Number of periods

    Returns:
        Future value
    """
    try:
        fv = present_value * ((1 + rate) ** periods)
        return f"Future Value: {fv:.2f} (FV = PV * (1 + r)^n)"
    except Exception as e:
        return f"Error calculating future value: {str(e)}"


async def calculate_econometric_regression(
    x_data: List[float],
    y_data: List[float],
) -> str:
    """
    Calculate simple linear regression for econometric analysis.

    Args:
        x_data: Independent variable data
        y_data: Dependent variable data

    Returns:
        Regression coefficients and statistics
    """
    try:
        if len(x_data) != len(y_data):
            return "Error: x_data and y_data must have equal length"

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

        return f"Regression: y = {slope:.6f}x + {intercept:.6f}\nR² = {r_squared:.6f}"
    except Exception as e:
        return f"Error in regression calculation: {str(e)}"


def get_business_law_tools() -> List[ToolDefinition]:
    """Get all business and law tool definitions."""
    return [
        ToolDefinition(
            name="calculate_financial_ratio",
            description="Calculate financial ratios: current_ratio, quick_ratio, debt_to_equity, profit_margin, roe (return on equity), roa (return on assets).",
            args_description={
                "ratio_type": "Type of ratio: current_ratio, quick_ratio, debt_to_equity, profit_margin, roe, roa",
                "current_assets": "Current assets",
                "current_liabilities": "Current liabilities",
                "total_debt": "Total debt",
                "total_equity": "Total equity",
                "net_income": "Net income",
                "revenue": "Revenue",
                "total_assets": "Total assets",
            },
            args_schema={
                "ratio_type": {"type": "string"},
                "current_assets": {"type": "number"},
                "current_liabilities": {"type": "number"},
                "total_debt": {"type": "number"},
                "total_equity": {"type": "number"},
                "net_income": {"type": "number"},
                "revenue": {"type": "number"},
                "total_assets": {"type": "number"},
            },
            tool=calculate_financial_ratio,
        ),
        ToolDefinition(
            name="calculate_npv",
            description="Calculate Net Present Value (NPV) of cash flows.",
            args_description={
                "cash_flows": "List of cash flows",
                "discount_rate": "Discount rate as decimal (e.g., 0.10 for 10%)",
                "initial_investment": "Initial investment (optional)",
            },
            args_schema={
                "cash_flows": {"type": "array", "items": {"type": "number"}},
                "discount_rate": {"type": "number"},
                "initial_investment": {"type": "number"},
            },
            tool=calculate_npv,
        ),
        ToolDefinition(
            name="calculate_irr",
            description="Calculate Internal Rate of Return (IRR) of cash flows.",
            args_description={
                "cash_flows": "List of cash flows (first should be negative for initial investment)",
            },
            args_schema={
                "cash_flows": {"type": "array", "items": {"type": "number"}},
            },
            tool=calculate_irr,
        ),
        ToolDefinition(
            name="calculate_present_value",
            description="Calculate present value using PV = FV / (1 + r)^n.",
            args_description={
                "future_value": "Future value",
                "rate": "Interest rate per period as decimal",
                "periods": "Number of periods",
            },
            args_schema={
                "future_value": {"type": "number"},
                "rate": {"type": "number"},
                "periods": {"type": "integer"},
            },
            tool=calculate_present_value,
        ),
        ToolDefinition(
            name="calculate_future_value",
            description="Calculate future value using FV = PV * (1 + r)^n.",
            args_description={
                "present_value": "Present value",
                "rate": "Interest rate per period as decimal",
                "periods": "Number of periods",
            },
            args_schema={
                "present_value": {"type": "number"},
                "rate": {"type": "number"},
                "periods": {"type": "integer"},
            },
            tool=calculate_future_value,
        ),
        ToolDefinition(
            name="calculate_econometric_regression",
            description="Calculate simple linear regression for econometric analysis.",
            args_description={
                "x_data": "Independent variable data",
                "y_data": "Dependent variable data",
            },
            args_schema={
                "x_data": {"type": "array", "items": {"type": "number"}},
                "y_data": {"type": "array", "items": {"type": "number"}},
            },
            tool=calculate_econometric_regression,
        ),
    ]
