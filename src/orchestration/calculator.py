"""
Calculator Tool (offline)

Safely evaluates a single math expression (no statements, no imports).
"""

import ast
import math
import operator as op
from typing import Any, Dict

from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger


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


def make_calculator_tool() -> ToolDefinition:
    return ToolDefinition(
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
    )
