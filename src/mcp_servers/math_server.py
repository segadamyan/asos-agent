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
from typing import Any, Dict, List, Optional

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
    """
    # Split on equals sign
    if "=" not in equation:
        raise ValueError("Equation must contain '='")
    
    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()
    
    # Try to solve by moving all to one side
    # For simple ax + b = c form
    
    # Parse left side
    var_pattern = rf"([+-]?\s*\d*\.?\d*)\s*\*?\s*{variable}"
    const_pattern = r"([+-]?\s*\d+\.?\d*)\s*(?![a-zA-Z])"
    
    # Extract coefficient of variable from left side
    var_match = re.search(var_pattern, left)
    left_coef = 1
    if var_match:
        coef_str = var_match.group(1).replace(" ", "")
        if coef_str in ["", "+"]:
            left_coef = 1
        elif coef_str == "-":
            left_coef = -1
        else:
            left_coef = float(coef_str)
    
    # Extract constant from left side
    left_without_var = re.sub(var_pattern, "", left)
    left_const = 0
    const_matches = re.findall(r"[+-]?\s*\d+\.?\d*", left_without_var)
    for match in const_matches:
        try:
            left_const += float(match.replace(" ", ""))
        except:
            pass
    
    # Parse right side (assuming it's just a number for now)
    try:
        right_val = float(right)
    except:
        raise ValueError(f"Right side must be a number, got: {right}")
    
    # Solve: coef * x + const = right_val
    # x = (right_val - const) / coef
    if left_coef == 0:
        raise ValueError("Coefficient of variable is 0, cannot solve")
    
    solution = (right_val - left_const) / left_coef
    
    return {
        "variable": variable,
        "value": solution,
        "equation": equation,
        "verification": f"{left_coef}*{solution} + {left_const} = {left_coef * solution + left_const}"
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
                        "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', '2^10')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="solve_equation",
            description="Solve a simple linear equation for a variable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "The equation to solve (e.g., '2x + 5 = 11')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to solve for (default: 'x')",
                        "default": "x"
                    }
                },
                "required": ["equation"]
            }
        ),
        Tool(
            name="factorial",
            description="Calculate the factorial of a non-negative integer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The non-negative integer to calculate factorial for"
                    }
                },
                "required": ["n"]
            }
        ),
        Tool(
            name="gcd",
            description="Calculate the greatest common divisor of two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="lcm",
            description="Calculate the least common multiple of two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="power",
            description="Calculate base raised to the power of exponent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "The base number"},
                    "exponent": {"type": "number", "description": "The exponent"}
                },
                "required": ["base", "exponent"]
            }
        ),
        Tool(
            name="sqrt",
            description="Calculate the square root of a number.",
            inputSchema={
                "type": "object",
                "properties": {
                    "number": {"type": "number", "description": "The number to calculate square root for"}
                },
                "required": ["number"]
            }
        ),
        Tool(
            name="statistics",
            description="Calculate statistical measures (mean, median, std_dev, variance, min, max, sum, range) for a list of numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to analyze"
                    }
                },
                "required": ["data"]
            }
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
                        "description": "First matrix (2D array)"
                    },
                    "matrix_b": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Second matrix (2D array)"
                    }
                },
                "required": ["matrix_a", "matrix_b"]
            }
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
                        "description": "Square matrix (2D array, up to 3x3)"
                    }
                },
                "required": ["matrix"]
            }
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
            result = base ** exponent
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
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

