"""
Python Executor Tool (offline)

Runs Python code in a restricted sandbox (no imports) with a timeout.
Captures printed output and returns a `result` variable if provided.
"""

import ast
import io
import math
import statistics
from contextlib import redirect_stdout
from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Dict

from agents.tools.base import ToolDefinition


class UnsafeCodeError(Exception):
    """Raised when code contains disallowed syntax or names."""


# Allow attribute access ONLY for these injected modules + members
_ALLOWED_ATTRS = {
    "math": {
        "factorial",
        "sqrt",
        "log",
        "log10",
        "exp",
        "sin",
        "cos",
        "tan",
        "pi",
        "e",
        "floor",
        "ceil",
        "comb",
        "perm",
        "gcd",
        "lcm",
    },
    "statistics": {
        "mean",
        "median",
        "pstdev",
        "stdev",
        "pvariance",
        "variance",
        "mode",
    },
}

_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Return,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.IfExp,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
    ast.boolop,

)

_BANNED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
)

_BANNED_NAMES = {
    "__import__",
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "help",
    "dir",
}


def _validate_code(code: str) -> ast.AST:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            raise UnsafeCodeError(f"Disallowed syntax: {node.__class__.__name__}")

        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeCodeError(f"Unsupported syntax: {node.__class__.__name__}")

        # Ban dangerous names
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise UnsafeCodeError(f"Disallowed name: {node.id}")

        # Ban calling dangerous names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
            raise UnsafeCodeError(f"Disallowed call: {node.func.id}")

        # Allow attributes only like: math.factorial, statistics.mean, etc.
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise UnsafeCodeError("Disallowed attribute access pattern.")
            base = node.value.id
            attr = node.attr
            if base not in _ALLOWED_ATTRS or attr not in _ALLOWED_ATTRS[base]:
                raise UnsafeCodeError(f"Disallowed attribute: {base}.{attr}")

    return tree


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    result: Any | None = None
    error: str | None = None


def _worker(code: str, q: "mp.Queue[ExecResult]") -> None:
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "range": range,
        "len": len,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "all": all,
        "any": any,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "print": print,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "statistics": statistics,
    }
    safe_locals: Dict[str, Any] = {}

    buf = io.StringIO()
    try:
        tree = _validate_code(code)
        with redirect_stdout(buf):
            exec(compile(tree, "<python_executor>", "exec"), safe_globals, safe_locals)
        q.put(ExecResult(ok=True, stdout=buf.getvalue(), result=safe_locals.get("result")))
    except Exception as e:
        q.put(ExecResult(ok=False, stdout=buf.getvalue(), error=f"{e.__class__.__name__}: {e}"))


async def python_executor(code: str, timeout_seconds: float = 2.0) -> Dict[str, Any]:
    src = (code or "").strip()
    if not src:
        return {"error": "Empty code."}

    # macOS: fork avoids spawn + <stdin> issues; fallback to spawn if fork not available
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(src, q))
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": f"Execution timed out after {timeout_seconds} seconds."}

    if q.empty():
        return {"error": "Execution failed: no result returned."}

    r: ExecResult = q.get()
    if r.ok:
        out: Dict[str, Any] = {"stdout": r.stdout}
        if r.result is not None:
            out["result"] = r.result
        return out

    return {"stdout": r.stdout, "error": r.error}


def make_python_executor_tool() -> ToolDefinition:
    return ToolDefinition(
        name="python_executor",
        description=(
            "Executes offline Python code safely for multi-step math/logic. "
            "No imports allowed. You can use `math` and `statistics`. "
            "Use `print()` for intermediate steps. Set `result = ...` to return a value."
        ),
        args_description={
            "code": "Python code to execute. Example: 'result = math.factorial(20)' or multi-line code.",
            "timeout_seconds": "Max execution time in seconds (default: 2.0).",
        },
        args_schema={
            "code": {"type": "string"},
            "timeout_seconds": {"type": "number"},
        },
        tool=python_executor,
    )
