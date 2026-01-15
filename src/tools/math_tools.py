"""
Mathematics Tools

"""

import ast
import math
import operator as op
from typing import Any, Dict, List, Optional

import sympy as sp

from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger
from tools.execution_tools import get_execution_tools

# ============================================================================
# Calculator Tool (offline, safe expression evaluator)
# ============================================================================

_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        fn = _ALLOWED_BIN_OPS.get(type(node.op))
        if fn is None:
            raise ValueError("Operator not allowed.")
        return fn(_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp):
        fn = _ALLOWED_UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError("Unary operator not allowed.")
        return fn(_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None or not callable(fn):
            raise ValueError(f"Function '{node.func.id}' is not allowed.")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")
        return fn(*[_eval(a) for a in node.args])

    raise ValueError(f"Unsupported syntax: {node.__class__.__name__}")


async def calculator(expression: str) -> Dict[str, Any]:
    """
    Safely evaluates a single math expression (no statements, no imports).

    Args:
        expression: Math expression string. Example: '2*(3+4)**2' or 'sqrt(16) + log10(100)'.

    Returns:
        Dict with "result" on success or "error" on failure.
    """
    logger.info(f"TOOL CALLED: calculator(expression='{expression}')")

    expr = (expression or "").strip()
    if not expr:
        return {"error": "Empty expression."}

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree.body)
        if isinstance(result, float):
            result = float(f"{result:.12g}")
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero."}
    except Exception as e:
        return {"error": f"Invalid expression ({e.__class__.__name__})."}


async def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")

    Returns:
        String representation of the result
    """
    try:
        # Create a safe evaluation context with common math functions
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "lcm": lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
            "degrees": math.degrees,
            "radians": math.radians,
        }

        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


async def statistical_calc(
    operation: str,
    data: List[float],
    percentile: Optional[float] = None,
    sample1: Optional[List[float]] = None,
    sample2: Optional[List[float]] = None,
) -> str:
    """
    Perform statistical calculations.

    Args:
        operation: Type of operation: mean, median, mode, std_dev, variance, min, max, range, percentile, correlation, t_test
        data: List of numbers for univariate statistics
        percentile: Percentile value (0-100) for percentile operation
        sample1: First sample for t-test or correlation
        sample2: Second sample for t-test or correlation

    Returns:
        Statistical result
    """
    try:
        operation = operation.lower()

        if operation in ["mean", "median", "mode", "std_dev", "variance", "min", "max", "range", "percentile"]:
            if not data:
                return "Error: No data provided"

            if operation == "mean":
                result = sum(data) / len(data)
                return f"Mean: {result:.6f}"

            elif operation == "median":
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    result = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
                else:
                    result = sorted_data[n // 2]
                return f"Median: {result:.6f}"

            elif operation == "mode":
                from collections import Counter

                counter = Counter(data)
                max_count = max(counter.values())
                modes = [k for k, v in counter.items() if v == max_count]
                return f"Mode: {modes}"

            elif operation == "std_dev":
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                result = math.sqrt(variance)
                return f"Standard Deviation: {result:.6f}"

            elif operation == "variance":
                mean = sum(data) / len(data)
                result = sum((x - mean) ** 2 for x in data) / len(data)
                return f"Variance: {result:.6f}"

            elif operation == "min":
                return f"Minimum: {min(data)}"

            elif operation == "max":
                return f"Maximum: {max(data)}"

            elif operation == "range":
                return f"Range: {max(data) - min(data)}"

            elif operation == "percentile":
                if percentile is None:
                    return "Error: percentile parameter required"
                sorted_data = sorted(data)
                index = (percentile / 100) * (len(sorted_data) - 1)
                if index.is_integer():
                    result = sorted_data[int(index)]
                else:
                    lower = sorted_data[int(index)]
                    upper = sorted_data[int(index) + 1]
                    result = lower + (upper - lower) * (index - int(index))
                return f"{percentile}th Percentile: {result:.6f}"

        elif operation == "correlation":
            if not sample1 or not sample2 or len(sample1) != len(sample2):
                return "Error: sample1 and sample2 must be provided and have equal length"
            n = len(sample1)
            mean1 = sum(sample1) / n
            mean2 = sum(sample2) / n
            numerator = sum((sample1[i] - mean1) * (sample2[i] - mean2) for i in range(n))
            denominator = math.sqrt(sum((x - mean1) ** 2 for x in sample1) * sum((y - mean2) ** 2 for y in sample2))
            result = numerator / denominator if denominator != 0 else 0
            return f"Correlation: {result:.6f}"

        elif operation == "t_test":
            if not sample1 or not sample2:
                return "Error: sample1 and sample2 must be provided"
            n1, n2 = len(sample1), len(sample2)
            mean1 = sum(sample1) / n1
            mean2 = sum(sample2) / n2
            var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1) if n1 > 1 else 0
            var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1) if n2 > 1 else 0
            pooled_std = math.sqrt((var1 / n1) + (var2 / n2))
            t_statistic = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
            return f"T-statistic: {t_statistic:.6f}, Mean1: {mean1:.6f}, Mean2: {mean2:.6f}"

        else:
            return f"Unknown operation: {operation}"

    except Exception as e:
        return f"Error in statistical calculation: {str(e)}"


async def solve_quadratic(a: float, b: float, c: float) -> str:
    """
    Solve a quadratic equation ax² + bx + c = 0.

    Args:
        a: Coefficient of x²
        b: Coefficient of x
        c: Constant term

    Returns:
        Solutions to the quadratic equation
    """
    try:
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            real_part = -b / (2 * a)
            imag_part = math.sqrt(-discriminant) / (2 * a)
            return f"Complex solutions: {real_part:.6f} ± {imag_part:.6f}i"
        elif discriminant == 0:
            solution = -b / (2 * a)
            return f"One solution: {solution:.6f}"
        else:
            solution1 = (-b + math.sqrt(discriminant)) / (2 * a)
            solution2 = (-b - math.sqrt(discriminant)) / (2 * a)
            return f"Solutions: x₁ = {solution1:.6f}, x₂ = {solution2:.6f}"
    except Exception as e:
        return f"Error solving quadratic: {str(e)}"


async def calculate_derivative(expression: str, variable: str = "x") -> str:
    """
    Calculate derivative of a mathematical expression using symbolic differentiation.

    Args:
        expression: Mathematical expression (e.g., "x^2 + 3*x + 5", "sin(x)*exp(x)")
        variable: Variable to differentiate with respect to (default: "x")

    Returns:
        Derivative expression as a string
    """
    try:
        # Convert string to sympy expression
        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Parse the expression
        var_symbol = sp.Symbol(variable)
        expr = sp.sympify(expression)

        # Calculate derivative
        derivative = sp.diff(expr, var_symbol)

        return f"d/d{variable}({expression}) = {derivative}"
    except Exception as e:
        return f"Error calculating derivative: {str(e)}"


async def calculate_integral(
    expression: str, variable: str = "x", lower: Optional[float] = None, upper: Optional[float] = None
) -> str:
    """
    Calculate integral of a mathematical expression using symbolic integration.

    Args:
        expression: Mathematical expression to integrate (e.g., "x^2 + 3*x + 5", "sin(x)*exp(x)")
        variable: Variable of integration (default: "x")
        lower: Lower bound for definite integral
        upper: Upper bound for definite integral

    Returns:
        Integral expression or numerical value for definite integrals
    """
    try:
        expression = expression.replace("^", "**")

        # Parse the expression
        var_symbol = sp.Symbol(variable)
        expr = sp.sympify(expression)

        # Calculate integral
        if lower is not None and upper is not None:
            # Definite integral
            integral_result = sp.integrate(expr, (var_symbol, lower, upper))
            # Try to evaluate numerically if possible
            try:
                numerical_value = float(integral_result.evalf())
                return f"∫[{lower}, {upper}] ({expression}) d{variable} = {integral_result} ≈ {numerical_value:.6f}"
            except:
                return f"∫[{lower}, {upper}] ({expression}) d{variable} = {integral_result}"
        else:
            # Indefinite integral
            integral_result = sp.integrate(expr, var_symbol)
            return f"∫({expression}) d{variable} = {integral_result} + C"
    except Exception as e:
        return f"Error calculating integral: {str(e)}"


def get_math_tools() -> List[ToolDefinition]:
    """Get all mathematics tool definitions."""
    tools = [
        ToolDefinition(
            name="calculator",
            description=(
                "Safely evaluates a single math expression. Supports +, -, *, /, //, %, **, "
                "parentheses, functions (sqrt, abs, round, floor, ceil, log, log10, exp, sin, cos, tan) "
                "and constants (pi, e)."
            ),
            args_description={
                "expression": "Math expression string. Example: '2*(3+4)**2' or 'sqrt(16) + log10(100)'."
            },
            args_schema={"expression": {"type": "string"}},
            tool=calculator,
        ),
        ToolDefinition(
            name="calculate",
            description="Evaluate a mathematical expression. Supports arithmetic, trigonometry, logarithms, and common math functions.",
            args_description={
                "expression": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
            },
            args_schema={
                "expression": {"type": "string"},
            },
            tool=calculate,
        ),
        ToolDefinition(
            name="statistical_calc",
            description="Perform statistical calculations: mean, median, mode, standard deviation, variance, min, max, range, percentile, correlation, t-test.",
            args_description={
                "operation": "Type of operation: mean, median, mode, std_dev, variance, min, max, range, percentile, correlation, t_test",
                "data": "List of numbers for univariate statistics",
                "percentile": "Percentile value (0-100) for percentile operation",
                "sample1": "First sample for correlation or t-test",
                "sample2": "Second sample for correlation or t-test",
            },
            args_schema={
                "operation": {"type": "string"},
                "data": {"type": "array", "items": {"type": "number"}},
                "percentile": {"type": "number", "minimum": 0, "maximum": 100},
                "sample1": {"type": "array", "items": {"type": "number"}},
                "sample2": {"type": "array", "items": {"type": "number"}},
            },
            tool=statistical_calc,
        ),
        ToolDefinition(
            name="solve_quadratic",
            description="Solve a quadratic equation ax² + bx + c = 0.",
            args_description={
                "a": "Coefficient of x²",
                "b": "Coefficient of x",
                "c": "Constant term",
            },
            args_schema={
                "a": {"type": "number"},
                "b": {"type": "number"},
                "c": {"type": "number"},
            },
            tool=solve_quadratic,
        ),
        ToolDefinition(
            name="calculate_derivative",
            description="Calculate derivative of a mathematical expression (basic implementation).",
            args_description={
                "expression": "Mathematical expression (e.g., 'x^2 + 3*x + 5')",
                "variable": "Variable to differentiate with respect to (default: 'x')",
            },
            args_schema={
                "expression": {"type": "string"},
                "variable": {"type": "string"},
            },
            tool=calculate_derivative,
        ),
        ToolDefinition(
            name="calculate_integral",
            description="Calculate integral of a mathematical expression (basic implementation).",
            args_description={
                "expression": "Mathematical expression to integrate",
                "variable": "Variable of integration (default: 'x')",
                "lower": "Lower bound for definite integral",
                "upper": "Upper bound for definite integral",
            },
            args_schema={
                "expression": {"type": "string"},
                "variable": {"type": "string"},
                "lower": {"type": "number"},
                "upper": {"type": "number"},
            },
            tool=calculate_integral,
        ),
    ]

    # Add python_executor from execution_tools
    tools.extend(get_execution_tools())

    return tools
