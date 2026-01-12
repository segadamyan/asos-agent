#!/usr/bin/env python3
"""
MCP Math Server

A Model Context Protocol server that provides mathematical tools.

Run standalone:
    python -m mcp_servers.math_server

Tools provided:
    - calculate: Basic arithmetic operations
    - solve_equation: Solve simple equations
    - factorial: Calculate factorial
    - gcd: Greatest common divisor
    - lcm: Least common multiple
    - power: Exponentiation
    - sqrt: Square root
    - statistics: Mean, median, std_dev, variance
    - matrix_multiply: Matrix multiplication
    - matrix_determinant: Matrix determinant (2x2, 3x3)
"""

import asyncio
import json
import math
import re
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server
server = Server("math-server")


def safe_eval_expression(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Supports: +, -, *, /, ^, **, (), numbers, and common math functions.
    """
    # Replace ^ with ** for exponentiation
    expression = expression.replace("^", "**")

    # Only allow safe characters and functions
    allowed_chars = set("0123456789+-*/().e ")
    allowed_funcs = ["sqrt", "sin", "cos", "tan", "log", "log10", "exp", "abs", "pi"]

    # Check for disallowed characters
    cleaned = expression
    for func in allowed_funcs:
        cleaned = cleaned.replace(func, "")
    cleaned = cleaned.replace("**", "")

    if not all(c in allowed_chars for c in cleaned):
        raise ValueError(f"Expression contains disallowed characters: {expression}")

    # Build safe namespace
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")


def solve_linear_equation(equation: str, variable: str = "x") -> Dict[str, Any]:
    """
    Solve a simple linear equation.

    Supports equations like:
        - "2x + 5 = 11"
        - "3*x - 7 = 8"
        - "x/2 = 10"
        - "2x + 3x = 10"
    """
    # Split on equals sign
    if "=" not in equation:
        raise ValueError("Equation must contain '='")

    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()

    # Helper to parse a side
    def parse_side(expression: str) -> tuple[float, float]:
        # Normalize: remove spaces around *, /, +, - to make regex simpler
        # But for now, let's just stick to a robust regex approach

        # We need to find all terms. A term is [+-]? [number]? [*]? [variable]?
        # But regexing global matches is tricky with overlapping.
        # Let's simplify: remove whitespace
        expr = expression.replace(" ", "")

        # Handle the case where expression starts without sign
        if not expr.startswith(("+", "-")):
            expr = "+" + expr

        # Pattern to match terms: ([+-]) (number)? ([*])? (variable)?
        # This is getting complex for regex.
        # Simpler approach: split by + or - (keeping delimiters)

        terms = re.findall(r"[+-][^+-]+", expr)

        total_coef = 0.0
        total_const = 0.0

        for term in terms:
            sign = 1 if term.startswith("+") else -1
            content = term[1:]  # remove sign

            if variable in content:
                # It's a variable term
                # Remove variable char
                coef_part = content.replace(variable, "")

                # Check for multiplication/division
                if "*" in coef_part:
                    coef_part = coef_part.replace("*", "")

                if coef_part == "":
                    val = 1.0
                elif coef_part == "/":  # handle x/2? No, x/2 would be /2
                    # content is "x/2" -> coef_part is "/2"
                    pass  # handled below
                else:
                    try:
                        val = float(coef_part)
                    except ValueError:
                        # Maybe it is /2
                        if coef_part.startswith("/"):
                            val = 1.0 / float(coef_part[1:])
                        else:
                            val = 1.0  # Fallback

                # Handle edge case like "x/2" where content was "x/2", coef_part became "/2"
                if content.endswith(f"/{variable}"):
                    # This logic handles linear only, so 1/x is not supported linear
                    pass
                elif f"/{variable}" in content:
                    # 1/x case
                    pass
                elif content.startswith(variable + "/"):
                    # x/2 case
                    denom = content.split("/")[1]
                    val = 1.0 / float(denom)

                total_coef += sign * val
            else:
                # It's a constant
                try:
                    total_const += sign * float(content)
                except ValueError:
                    pass  # Ignore unparseable

        return total_coef, total_const

    # Parse both sides
    # Note: The original regex-based approach was fragile.
    # A cleaner approach without full parser is to rely on simple term splitting.

    # Let's use a simpler logic for this specific tool:
    # 1. Move everything to left side: left - right = 0
    # 2. Iterate through terms

    # Combine left and right (invert right signs)
    # Actually, let's reuse the logic but be careful.

    # IMPROVED LOGIC:
    # 1. Normalize spaces
    # 2. Find all matches of `[+-]? \d* \.? \d* x` for variable
    # 3. Find all matches of `[+-]? \d+ \.? \d*` for constants (careful not to double count)

    # Let's try to match variable terms first, remove them, then match constants.

    def parse_poly_linear(side_str: str) -> tuple[float, float]:
        # Remove spaces
        s = side_str.replace(" ", "")
        if not s:
            return 0.0, 0.0

        # Add + if no sign at start
        if s[0] not in "+-":
            s = "+" + s

        # Find variable terms: [+-] [float]? [*]? x [/]? [float]?
        # We assume linear: ax, x/a, x. Not a/x.

        # Regex for variable term
        # Groups: 1=Sign, 2=Coef, 3=Divisor
        # Examples: +2x, -x, +x/2, -3.5*x

        coef_sum = 0.0

        # Find all variable terms
        # This regex looks for [+-], optional number, optional *, var, optional / number
        # We mask them out after finding to avoid matching constants later

        var_terms = re.finditer(rf"([+-])(\d*\.?\d*)?\*?{variable}(?:/(\d+\.?\d*))?", s)

        last_end = 0
        masked_s = list(s)  # To mask out used chars

        for match in var_terms:
            start, end = match.span()
            sign_str = match.group(1)
            coef_str = match.group(2)
            denom_str = match.group(3)

            # Determine value
            val = 1.0
            if coef_str:
                val = float(coef_str)

            if denom_str:
                val /= float(denom_str)

            if sign_str == "-":
                val = -val

            coef_sum += val

            # Mask this part in string so constant finder ignores it
            for i in range(start, end):
                masked_s[i] = " "

        # Now find constants in the remaining string
        const_sum = 0.0
        remaining = "".join(masked_s).replace(" ", "")

        # If we created fragments like "+ +5", fix it.
        # Actually simplest is to just re-scan for numbers with signs.
        # But we might have destroyed the context.
        # Simpler: The variable regex consumes the sign.
        # Any remaining part is `[+-] number`.

        # Let's just find numbers with signs in the original string that WEREN'T part of variable terms.
        # The masking approach is safe.

        # remaining is like "+5-3".
        # We need to ensure we don't merge signs if masking left holes.
        # Actually, `re.findall(r'[+-]?\d+\.?\d*', remaining)` might be risky if we left bare numbers.
        # But we consumed the sign with the variable term.
        # So "+2x+5" -> "   +5". Correct.
        # "5+2x" -> "+5+2x" -> "+5   ". Correct.

        # Let's just parse the masked string for constants
        # We need to handle the fact that we might have "separated" signs from numbers?
        # No, because the variable regex includes the leading [+-].

        const_matches = re.finditer(r"([+-])?(\d+\.?\d*)", remaining)
        for match in const_matches:
            sign_str = match.group(1)
            val_str = match.group(2)

            val = float(val_str)
            if sign_str == "-":
                val = -val

            const_sum += val

        return coef_sum, const_sum

    left_coef, left_const = parse_poly_linear(left)
    right_coef, right_const = parse_poly_linear(right)

    # Equation: left_coef*x + left_const = right_coef*x + right_const
    # (left_coef - right_coef)*x = right_const - left_const

    final_coef = left_coef - right_coef
    final_const = right_const - left_const

    if abs(final_coef) < 1e-10:
        if abs(final_const) < 1e-10:
            raise ValueError("Identity equation (0=0), infinite solutions")
        else:
            raise ValueError("Contradiction (0=k), no solution")

    solution = final_const / final_coef

    return {
        "variable": variable,
        "value": solution,
        "equation": equation,
        "verification": f"{final_coef}*{solution} = {final_const}",
    }


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a dataset."""
    n = len(data)
    if n == 0:
        raise ValueError("Data list cannot be empty")

    # Mean
    mean = sum(data) / n

    # Median
    sorted_data = sorted(data)
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

    # Variance and Standard Deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)

    # Min and Max
    min_val = min(data)
    max_val = max(data)

    return {
        "count": n,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "variance": round(variance, 6),
        "std_dev": round(std_dev, 6),
        "min": min_val,
        "max": max_val,
        "sum": sum(data),
        "range": max_val - min_val,
    }


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply: A is {rows_a}x{cols_a}, B is {rows_b}x{cols_b}")

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result


def matrix_determinant(matrix: List[List[float]]) -> float:
    """Calculate the determinant of a square matrix (up to 3x3)."""
    n = len(matrix)

    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif n == 3:
        return (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
    else:
        raise ValueError("Determinant calculation only supported for matrices up to 3x3")


# Register tools with the MCP server


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available math tools."""
    return [
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression. Supports +, -, *, /, ^, (), sqrt, sin, cos, tan, log, exp, abs, pi, e.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', '2^10')",
                    }
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="solve_equation",
            description="Solve a simple linear equation for a variable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "The equation to solve (e.g., '2x + 5 = 11')"},
                    "variable": {
                        "type": "string",
                        "description": "The variable to solve for (default: 'x')",
                        "default": "x",
                    },
                },
                "required": ["equation"],
            },
        ),
        Tool(
            name="factorial",
            description="Calculate the factorial of a non-negative integer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "The non-negative integer to calculate factorial for"}
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="gcd",
            description="Calculate the greatest common divisor of two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="lcm",
            description="Calculate the least common multiple of two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="power",
            description="Calculate base raised to the power of exponent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "The base number"},
                    "exponent": {"type": "number", "description": "The exponent"},
                },
                "required": ["base", "exponent"],
            },
        ),
        Tool(
            name="sqrt",
            description="Calculate the square root of a number.",
            inputSchema={
                "type": "object",
                "properties": {"number": {"type": "number", "description": "The number to calculate square root for"}},
                "required": ["number"],
            },
        ),
        Tool(
            name="statistics",
            description="Calculate statistical measures (mean, median, std_dev, variance, min, max, sum, range) for a list of numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}, "description": "List of numbers to analyze"}
                },
                "required": ["data"],
            },
        ),
        Tool(
            name="matrix_multiply",
            description="Multiply two matrices.",
            inputSchema={
                "type": "object",
                "properties": {
                    "matrix_a": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "First matrix (2D array)",
                    },
                    "matrix_b": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Second matrix (2D array)",
                    },
                },
                "required": ["matrix_a", "matrix_b"],
            },
        ),
        Tool(
            name="matrix_determinant",
            description="Calculate the determinant of a square matrix (up to 3x3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Square matrix (2D array, up to 3x3)",
                    }
                },
                "required": ["matrix"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "calculate":
            expression = arguments.get("expression", "")
            result = safe_eval_expression(expression)
            return [TextContent(type="text", text=f"Result: {result}")]

        elif name == "solve_equation":
            equation = arguments.get("equation", "")
            variable = arguments.get("variable", "x")
            result = solve_linear_equation(equation, variable)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "factorial":
            n = arguments.get("n", 0)
            if n < 0:
                return [TextContent(type="text", text="Error: Factorial not defined for negative numbers")]
            result = math.factorial(n)
            return [TextContent(type="text", text=f"{n}! = {result}")]

        elif name == "gcd":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result = math.gcd(a, b)
            return [TextContent(type="text", text=f"GCD({a}, {b}) = {result}")]

        elif name == "lcm":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result = math.lcm(a, b)
            return [TextContent(type="text", text=f"LCM({a}, {b}) = {result}")]

        elif name == "power":
            base = arguments.get("base", 0)
            exponent = arguments.get("exponent", 0)
            result = base**exponent
            return [TextContent(type="text", text=f"{base}^{exponent} = {result}")]

        elif name == "sqrt":
            number = arguments.get("number", 0)
            if number < 0:
                return [TextContent(type="text", text="Error: Cannot calculate square root of negative number")]
            result = math.sqrt(number)
            return [TextContent(type="text", text=f"√{number} = {result}")]

        elif name == "statistics":
            data = arguments.get("data", [])
            if not data:
                return [TextContent(type="text", text="Error: Data list cannot be empty")]
            result = calculate_statistics(data)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "matrix_multiply":
            matrix_a = arguments.get("matrix_a", [])
            matrix_b = arguments.get("matrix_b", [])
            result = matrix_multiply(matrix_a, matrix_b)
            return [TextContent(type="text", text=f"Result:\n{json.dumps(result, indent=2)}")]

        elif name == "matrix_determinant":
            matrix = arguments.get("matrix", [])
            result = matrix_determinant(matrix)
            return [TextContent(type="text", text=f"Determinant = {result}")]

        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
