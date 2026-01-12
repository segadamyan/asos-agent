#!/usr/bin/env python3
"""
MCP Symbolic Math Server

An advanced Model Context Protocol server that provides symbolic mathematics
capabilities using SymPy, plus numerical computation via NumPy.

This serves as an "external-like" advanced math server with capabilities
beyond the basic math_server.py.

Run standalone:
    python -m mcp_servers.symbolic_math_server

Tools provided:
    - symbolic_solve: Solve equations symbolically
    - symbolic_simplify: Simplify algebraic expressions
    - symbolic_differentiate: Compute derivatives
    - symbolic_integrate: Compute integrals (definite and indefinite)
    - symbolic_limit: Compute limits
    - symbolic_series: Taylor/Maclaurin series expansion
    - symbolic_factor: Factor polynomials
    - symbolic_expand: Expand expressions
    - matrix_inverse: Compute matrix inverse
    - matrix_eigenvalues: Compute eigenvalues and eigenvectors
    - polynomial_roots: Find roots of polynomials
    - trigonometric_simplify: Simplify trigonometric expressions
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import sympy as sp
from sympy import (
    Symbol, symbols, solve, simplify, diff, integrate,
    limit, series, factor, expand, trigsimp, sqrt, sin, cos, tan,
    log, exp, pi, E, I, oo, Matrix, Rational, nsimplify
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


# Create the MCP server
server = Server("symbolic-math-server")

# Parser transformations for more natural input
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def safe_parse(expr_str: str, local_dict: Optional[Dict] = None) -> sp.Expr:
    """
    Safely parse a mathematical expression string into a SymPy expression.
    """
    if local_dict is None:
        local_dict = {}
    
    # Add common symbols
    x, y, z, t, n, a, b, c = symbols('x y z t n a b c')
    default_symbols = {'x': x, 'y': y, 'z': z, 't': t, 'n': n, 'a': a, 'b': b, 'c': c,
                       'pi': pi, 'e': E, 'i': I, 'inf': oo, 'sqrt': sqrt,
                       'sin': sin, 'cos': cos, 'tan': tan, 'log': log, 'exp': exp}
    default_symbols.update(local_dict)
    
    try:
        return parse_expr(expr_str, local_dict=default_symbols, transformations=TRANSFORMATIONS)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr_str}': {e}")


def format_sympy_result(result: Any) -> str:
    """Format a SymPy result for display."""
    if isinstance(result, list):
        return "[" + ", ".join(str(r) for r in result) + "]"
    elif isinstance(result, dict):
        return json.dumps({str(k): str(v) for k, v in result.items()}, indent=2)
    elif isinstance(result, Matrix):
        return str(result)
    else:
        return str(result)


# Register tools

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available symbolic math tools."""
    return [
        Tool(
            name="symbolic_solve",
            description="Solve equations symbolically. Can solve single equations or systems of equations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of equations to solve (e.g., ['x**2 - 4', '2*x + y = 5']). Use '=' for equations or expressions equal to 0."
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to solve for (e.g., ['x', 'y'])"
                    }
                },
                "required": ["equations", "variables"]
            }
        ),
        Tool(
            name="symbolic_simplify",
            description="Simplify an algebraic expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to simplify (e.g., '(x**2 - 1)/(x - 1)')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="symbolic_differentiate",
            description="Compute the derivative of an expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to differentiate (e.g., 'x**3 + 2*x')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to differentiate with respect to (default: 'x')",
                        "default": "x"
                    },
                    "order": {
                        "type": "integer",
                        "description": "Order of derivative (default: 1)",
                        "default": 1
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="symbolic_integrate",
            description="Compute the integral of an expression (definite or indefinite).",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to integrate (e.g., 'x**2')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to integrate with respect to (default: 'x')",
                        "default": "x"
                    },
                    "lower_bound": {
                        "type": "string",
                        "description": "Lower bound for definite integral (optional, e.g., '0' or '-inf')"
                    },
                    "upper_bound": {
                        "type": "string",
                        "description": "Upper bound for definite integral (optional, e.g., '1' or 'inf')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="symbolic_limit",
            description="Compute the limit of an expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression (e.g., 'sin(x)/x')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable (default: 'x')",
                        "default": "x"
                    },
                    "point": {
                        "type": "string",
                        "description": "The point to take limit at (e.g., '0', 'inf', '-inf')"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["+", "-", "+-"],
                        "description": "Direction of limit: '+' (right), '-' (left), '+-' (both)",
                        "default": "+-"
                    }
                },
                "required": ["expression", "point"]
            }
        ),
        Tool(
            name="symbolic_series",
            description="Compute Taylor/Maclaurin series expansion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to expand (e.g., 'exp(x)', 'sin(x)')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable (default: 'x')",
                        "default": "x"
                    },
                    "point": {
                        "type": "string",
                        "description": "Point to expand around (default: '0' for Maclaurin)",
                        "default": "0"
                    },
                    "order": {
                        "type": "integer",
                        "description": "Number of terms (default: 6)",
                        "default": 6
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="symbolic_factor",
            description="Factor a polynomial expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The polynomial to factor (e.g., 'x**2 - 4')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="symbolic_expand",
            description="Expand an expression (opposite of factor).",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to expand (e.g., '(x + 1)**3')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="matrix_inverse",
            description="Compute the inverse of a square matrix.",
            inputSchema={
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Square matrix as 2D array"
                    }
                },
                "required": ["matrix"]
            }
        ),
        Tool(
            name="matrix_eigenvalues",
            description="Compute eigenvalues and eigenvectors of a square matrix.",
            inputSchema={
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Square matrix as 2D array"
                    }
                },
                "required": ["matrix"]
            }
        ),
        Tool(
            name="polynomial_roots",
            description="Find all roots (zeros) of a polynomial.",
            inputSchema={
                "type": "object",
                "properties": {
                    "coefficients": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Polynomial coefficients from highest to lowest degree (e.g., [1, 0, -4] for x² - 4)"
                    }
                },
                "required": ["coefficients"]
            }
        ),
        Tool(
            name="trigonometric_simplify",
            description="Simplify trigonometric expressions using identities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The trigonometric expression (e.g., 'sin(x)**2 + cos(x)**2')"
                    }
                },
                "required": ["expression"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "symbolic_solve":
            equations_str = arguments.get("equations", [])
            variables_str = arguments.get("variables", [])
            
            # Parse variables
            var_symbols = symbols(' '.join(variables_str))
            if not isinstance(var_symbols, tuple):
                var_symbols = (var_symbols,)
            var_dict = {str(v): v for v in var_symbols}
            
            # Parse equations
            equations = []
            for eq_str in equations_str:
                if '=' in eq_str:
                    left, right = eq_str.split('=', 1)
                    eq = safe_parse(left.strip(), var_dict) - safe_parse(right.strip(), var_dict)
                else:
                    eq = safe_parse(eq_str, var_dict)
                equations.append(eq)
            
            result = solve(equations, var_symbols)
            return [TextContent(type="text", text=f"Solutions: {format_sympy_result(result)}")]
        
        elif name == "symbolic_simplify":
            expr = safe_parse(arguments.get("expression", ""))
            result = simplify(expr)
            return [TextContent(type="text", text=f"Simplified: {result}")]
        
        elif name == "symbolic_differentiate":
            expr = safe_parse(arguments.get("expression", ""))
            var = Symbol(arguments.get("variable", "x"))
            order = arguments.get("order", 1)
            result = diff(expr, var, order)
            return [TextContent(type="text", text=f"Derivative: {result}")]
        
        elif name == "symbolic_integrate":
            expr = safe_parse(arguments.get("expression", ""))
            var = Symbol(arguments.get("variable", "x"))
            lower = arguments.get("lower_bound")
            upper = arguments.get("upper_bound")
            
            if lower is not None and upper is not None:
                lower_val = safe_parse(lower)
                upper_val = safe_parse(upper)
                result = integrate(expr, (var, lower_val, upper_val))
                return [TextContent(type="text", text=f"Definite integral from {lower} to {upper}: {result}")]
            else:
                result = integrate(expr, var)
                return [TextContent(type="text", text=f"Indefinite integral: {result} + C")]
        
        elif name == "symbolic_limit":
            expr = safe_parse(arguments.get("expression", ""))
            var = Symbol(arguments.get("variable", "x"))
            point = safe_parse(arguments.get("point", "0"))
            direction = arguments.get("direction", "+-")
            
            if direction == "+-":
                # Check both sides to be sure
                left_lim = limit(expr, var, point, dir="-")
                right_lim = limit(expr, var, point, dir="+")
                
                if left_lim == right_lim:
                    result = left_lim
                else:
                    return [TextContent(type="text", text=f"Limit does not exist (Left: {left_lim}, Right: {right_lim})")]
            else:
                result = limit(expr, var, point, dir=direction)
            return [TextContent(type="text", text=f"Limit as {var} → {point}: {result}")]
        
        elif name == "symbolic_series":
            expr = safe_parse(arguments.get("expression", ""))
            var = Symbol(arguments.get("variable", "x"))
            point = safe_parse(arguments.get("point", "0"))
            order = arguments.get("order", 6)
            result = series(expr, var, point, order)
            return [TextContent(type="text", text=f"Series expansion: {result}")]
        
        elif name == "symbolic_factor":
            expr = safe_parse(arguments.get("expression", ""))
            result = factor(expr)
            return [TextContent(type="text", text=f"Factored: {result}")]
        
        elif name == "symbolic_expand":
            expr = safe_parse(arguments.get("expression", ""))
            result = expand(expr)
            return [TextContent(type="text", text=f"Expanded: {result}")]
        
        elif name == "matrix_inverse":
            matrix_data = arguments.get("matrix", [])
            m = Matrix(matrix_data)
            if m.det() == 0:
                return [TextContent(type="text", text="Error: Matrix is singular (determinant = 0), no inverse exists")]
            result = m.inv()
            # Convert to float for nicer display
            result_float = [[float(x) if x.is_number else str(x) for x in row] for row in result.tolist()]
            return [TextContent(type="text", text=f"Inverse matrix:\n{json.dumps(result_float, indent=2)}")]
        
        elif name == "matrix_eigenvalues":
            matrix_data = arguments.get("matrix", [])
            m = Matrix(matrix_data)
            eigenvals = m.eigenvals()
            eigenvects = m.eigenvects()
            
            result = {
                "eigenvalues": {str(k): int(v) for k, v in eigenvals.items()},
                "eigenvectors": [
                    {
                        "eigenvalue": str(ev[0]),
                        "multiplicity": ev[1],
                        "vectors": [str(v) for v in ev[2]]
                    }
                    for ev in eigenvects
                ]
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "polynomial_roots":
            coefficients = arguments.get("coefficients", [])
            roots = np.roots(coefficients)
            # Format complex roots nicely
            formatted_roots = []
            for r in roots:
                if np.isreal(r):
                    formatted_roots.append(float(np.real(r)))
                else:
                    formatted_roots.append(f"{np.real(r):.6f} + {np.imag(r):.6f}i")
            return [TextContent(type="text", text=f"Roots: {formatted_roots}")]
        
        elif name == "trigonometric_simplify":
            expr = safe_parse(arguments.get("expression", ""))
            result = trigsimp(expr)
            return [TextContent(type="text", text=f"Simplified: {result}")]
        
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




