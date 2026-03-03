"""Evaluator that walks an expression AST and produces an :class:`~viewer.expression.Expression`."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .expression import Expression
from .expression_node import (
    BinaryOp,
    BinaryOpNode,
    ExprNode,
    FunctionCallNode,
    NumberNode,
    UnaryOp,
    UnaryOpNode,
    VariableRefNode,
)
from .variable import Variable

# ---------------------------------------------------------------------------
# Supported mathematical functions
# ---------------------------------------------------------------------------

# Maps lower-cased function name → callable that accepts an ndarray and
# returns an ndarray.  All functions must handle complex inputs gracefully
# because AC analysis variables are complex-valued.
_FUNCTION_IMPLS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs":    np.abs,
    "sqrt":   np.sqrt,
    "log":    np.log10,   # SPICE convention: log ≡ log10
    "log10":  np.log10,
    "ln":     np.log,
    "db":     lambda x: 20.0 * np.log10(np.abs(x)),
    "real":   np.real,
    "imag":   np.imag,
    "angle":  lambda x: np.angle(x, deg=True),
    "mag":    np.abs,
    "sin":    np.sin,
    "cos":    np.cos,
    "tan":    np.tan,
    "exp":    np.exp,
}

# ---------------------------------------------------------------------------
# Unit propagation helpers
# ---------------------------------------------------------------------------

def _propagate_binary_unit(left_unit: str, op: BinaryOp, right_unit: str) -> str:
    """Infer the resulting physical unit of a binary operation."""
    if op in (BinaryOp.ADD, BinaryOp.SUB):
        # units must be identical for addition/subtraction
        return left_unit if left_unit == right_unit else ""
    if op == BinaryOp.MUL:
        if {left_unit, right_unit} == {"V", "A"}:
            return "W"
        # scalar (dimensionless) factor preserves the other operand's unit
        if left_unit == "":
            return right_unit
        if right_unit == "":
            return left_unit
        return ""
    if op == BinaryOp.DIV:
        if left_unit == right_unit:
            return ""  # dimensionless ratio
        if left_unit == "V" and right_unit == "A":
            return "Ω"
        if left_unit == "A" and right_unit == "V":
            return "S"
        if right_unit == "":
            return left_unit  # divide by scalar
        return ""
    # POW — unit tracking for arbitrary exponents is not well-defined
    return ""


def _function_unit(func_name: str, arg_unit: str) -> str:
    """Infer the unit produced by a mathematical function call."""
    lower = func_name.lower()
    # functions that produce a fixed unit regardless of input
    if lower == "db":
        return "dB"
    if lower == "angle":
        return "°"
    # functions that preserve the physical unit of their argument
    if lower in ("abs", "real", "imag", "mag"):
        return arg_unit
    # all other functions (sqrt, log, sin, exp, …) strip the physical dimension
    return ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ExpressionEvaluator:
    """Stateless service that walks an expression AST and returns an
    :class:`~viewer.expression.Expression`.

    Usage example::

        parser    = ExpressionParser()
        evaluator = ExpressionEvaluator()

        tree    = parser.parse("V(R1) / I(R1)")
        context = {"V(R1)": var_vr1, "I(R1)": var_ir1}
        result  = evaluator.evaluate(tree, context, name="Z(R1)")
        # result.unit == "Ω"
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        node: ExprNode,
        context: dict[str, Variable],
        name: str | None = None,
        source: str | None = None,
    ) -> Expression:
        """Evaluate *node* against *context* and return an :class:`~viewer.expression.Expression`.

        :param node:    Root AST node returned by :class:`~viewer.expression_parser.ExpressionParser`.
        :param context: Mapping from variable name to :class:`~viewer.variable.Variable`.
                        Lookup is case-insensitive.
        :param name:    Display name for the resulting ``Expression``.  Defaults to
                        a string reconstruction of the AST.
        :param source:  Original source expression string stored on the result for
                        display / debugging.  Defaults to *name*.
        :raises ValueError: If a variable is not found in *context* or an unknown
                            function is encountered.
        """
        data, unit = self._eval(node, context)
        expr_name = name if name is not None else self._node_to_str(node)
        return Expression(expr_name, data, unit, source=source if source is not None else expr_name)

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _eval(self, node: ExprNode, context: dict[str, Variable]) -> tuple[np.ndarray, str]:
        """Recursively evaluate *node*, returning ``(data_array, unit_string)``."""
        if isinstance(node, NumberNode):
            return np.array(node.value), ""

        if isinstance(node, VariableRefNode):
            var = self._lookup(node.name, context)
            return var.values, var.type.value.unit

        if isinstance(node, FunctionCallNode):
            return self._eval_function(node, context)

        if isinstance(node, BinaryOpNode):
            left_data, left_unit = self._eval(node.left, context)
            right_data, right_unit = self._eval(node.right, context)
            data = self._apply_binary_op(node.op, left_data, right_data)
            unit = _propagate_binary_unit(left_unit, node.op, right_unit)
            return data, unit

        if isinstance(node, UnaryOpNode):
            data, unit = self._eval(node.operand, context)
            if node.op == UnaryOp.NEG:
                return -data, unit
            raise ValueError(f"unsupported unary operator: {node.op}")

        raise ValueError(f"unknown AST node type: {type(node).__name__}")

    def _eval_function(
        self, node: FunctionCallNode, context: dict[str, Variable]
    ) -> tuple[np.ndarray, str]:
        func = _FUNCTION_IMPLS.get(node.name.lower())
        if func is None:
            raise ValueError(f"unknown function: {node.name!r}")
        if len(node.args) != 1:
            raise ValueError(
                f"function {node.name!r} expects exactly 1 argument, got {len(node.args)}"
            )
        arg_data, arg_unit = self._eval(node.args[0], context)
        result_data = func(arg_data)
        result_unit = _function_unit(node.name, arg_unit)
        return result_data, result_unit

    @staticmethod
    def _apply_binary_op(op: BinaryOp, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if op == BinaryOp.ADD:
            return left + right
        if op == BinaryOp.SUB:
            return left - right
        if op == BinaryOp.MUL:
            return left * right
        if op == BinaryOp.DIV:
            return left / right
        if op == BinaryOp.POW:
            return left ** right
        raise ValueError(f"unsupported binary operator: {op}")

    @staticmethod
    def _lookup(name: str, context: dict[str, Variable]) -> Variable:
        """Look up *name* in *context* with case-insensitive fallback."""
        var = context.get(name)
        if var is not None:
            return var
        lower = name.lower()
        for key, v in context.items():
            if key.lower() == lower:
                return v
        raise ValueError(f"undefined variable: {name!r}")

    # ------------------------------------------------------------------
    # AST → string reconstruction (used for default expression names)
    # ------------------------------------------------------------------

    def _node_to_str(self, node: ExprNode) -> str:
        """Reconstruct a human-readable string from an AST node."""
        if isinstance(node, NumberNode):
            v = node.value
            return str(int(v)) if v == int(v) else str(v)
        if isinstance(node, VariableRefNode):
            return node.name
        if isinstance(node, FunctionCallNode):
            args_str = ", ".join(self._node_to_str(a) for a in node.args)
            return f"{node.name}({args_str})"
        if isinstance(node, BinaryOpNode):
            left = self._node_to_str(node.left)
            right = self._node_to_str(node.right)
            return f"({left} {node.op.value} {right})"
        if isinstance(node, UnaryOpNode):
            return f"-{self._node_to_str(node.operand)}"
        raise ValueError(f"unknown AST node type: {type(node).__name__}")
