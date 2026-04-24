from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from .expression import Expression
from .expression_node import BinaryOperator, BinaryOperatorNode, ExpressionNode, FunctionCallNode, NumberNode, UnaryOperator, UnaryOperatorNode, VariableRefNode

# maps lower-cased function name to a callable that accepts an ndarray and
# returns an ndarray; all functions must handle complex inputs gracefully
# because AC analysis variables are complex-valued
_FUNCTION_IMPLS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    # in SPICE, log is an alias for log10
    "log": np.log10,
    "log10": np.log10,
    "ln": np.log,
    "db": lambda x: 20.0 * np.log10(np.abs(x)),
    "real": np.real,
    "imag": np.imag,
    "angle": lambda x: np.angle(x, deg=True),
    # ph / phase are QSPICE aliases for angle (returns degrees)
    "ph": lambda x: np.angle(x, deg=True),
    "phase": lambda x: np.angle(x, deg=True),
    "mag": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    # inverse trigonometric functions; return values are in radians
    "asin": np.arcsin,
    "arcsin": np.arcsin,
    "acos": np.arccos,
    "arccos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,
    # hyperbolic functions
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    # complex conjugate; preserves the physical unit of its argument
    "conj": np.conj,
    # sqr(x) = x^2; unit is not tracked (would be unit-squared)
    "sqr": lambda x: x ** 2,
    # sign / sgn: −1, 0 or +1 element-wise; dimensionless
    "sign": np.sign,
    "sgn": np.sign,
    # uramp(x) = max(x, 0); used in SPICE behavioural sources
    "uramp": lambda x: np.maximum(np.real(x), 0.0),
    # rounding functions; preserve the physical unit
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    # int(x) truncates toward zero, matching SPICE/QSPICE behaviour
    "int": np.trunc,
}


# two-argument functions: (a, b) → ndarray
_FUNCTION_IMPLS_2: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    # atan2(y, x) follows the standard C / numpy argument order; returns radians
    "atan2": lambda y, x: np.arctan2(np.real(y), np.real(x)),
    # pow(x, y) = x^y — function form of the ^ operator
    "pow": lambda x, y: x ** y,
    # pwr(x, y) = |x|^y — always non-negative (SPICE convention for pwr)
    "pwr": lambda x, y: np.abs(x) ** np.real(y),
    # pwrs(x, y) = sgn(x) * |x|^y — signed power
    "pwrs": lambda x, y: np.sign(np.real(x)) * np.abs(x) ** np.real(y),
    # min / max operate on the real parts of their arguments
    "min": lambda a, b: np.minimum(np.real(a), np.real(b)),
    "max": lambda a, b: np.maximum(np.real(a), np.real(b)),
}


# three-argument functions: (x, lo, hi) → ndarray
_FUNCTION_IMPLS_3: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    # limit(x, lo, hi) clamps x; equivalent to LTspice / QSPICE limit()
    "limit": lambda x, lo, hi: np.clip(np.real(x), np.real(lo), np.real(hi)),
}


# matches the two-node differential probe form produced by the parser, e.g. "V(a, b)" or
# "V(net-foo, 0)"; group 1 = function name, group 2 = first node, group 3 = second node.
# Node names can contain any character except ',' and ')' (bullets, slashes, hashes, etc.).
_TWO_ARG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(([^,]+),\s*([^)]+)\)$")

# built-in numeric constants recognized as named identifiers in SPICE/QSPICE expressions.
# Each entry is a (value, unit) pair so that unit propagation works correctly.
# "pi" is a dimensionless mathematical constant; "mho" is the conductance unit (S = A/V) used
# in QSPICE-generated .alias expressions such as '.alias I(R4) (1mho*V(out,0))'.
_CONSTANTS: dict[str, tuple[float, str]] = {
    "pi": (np.pi, ""),
    "mho": (1.0, "S"),
    "s": (1.0, "s"),
}


def _propagate_binary_unit(left_unit: str, op: BinaryOperator, right_unit: str) -> str:
    # +, -
    if op in (BinaryOperator.ADD, BinaryOperator.SUB):
        # units must be identical for addition/subtraction
        return left_unit if left_unit == right_unit else ""
    # *
    if op == BinaryOperator.MUL:
        # W = V × A
        if {left_unit, right_unit} == {"V", "A"}:
            return "W"
        # S = A × V  (same as V × A, but check both orders)
        if {left_unit, right_unit} == {"S", "V"}:
            # siemens × volt = ampere  (S = A/V  →  S·V = A)
            return "A"
        # if one side is dimensionless, the result has the same unit as the other side
        if left_unit == "":
            return right_unit
        # left unit if any
        return left_unit
    # /
    if op == BinaryOperator.DIV:
        # unit / unit = dimensionless ratio
        if left_unit == right_unit:
            return ""
        # Ω = V / A
        if left_unit == "V" and right_unit == "A":
            return "Ω"
        # S = A / V
        if left_unit == "A" and right_unit == "V":
            return "S"
        # if the denominator is dimensionless, the result has the same unit as the numerator
        if right_unit == "":
            return left_unit
        # if the numerator is dimensionless
        if left_unit == "":
            # Ω = 1 / S
            if right_unit == "S":
                return "Ω"
            # S = 1 / Ω
            if right_unit == "Ω":
                return "S"
            # Hz = 1 / s
            if right_unit == "s":
                return "Hz"
            # s = 1 / Hz
            if right_unit == "Hz":
                return "s"
        # reset unit
        return ""
    # pow — unit tracking for arbitrary exponents is not well-defined
    return ""


def _function_unit(func_name: str, arg_unit: str) -> str:
    """Infer the unit produced by a single-argument mathematical function call."""
    lower = func_name.lower()
    # always dB
    if lower == "db":
        return "dB"
    # always angle in degrees
    if lower in ("angle", "ph", "phase"):
        return "°"
    # functions that preserve the physical unit of their argument
    if lower in ("abs", "real", "imag", "mag", "conj", "uramp", "round", "floor", "ceil", "int"):
        return arg_unit
    # all other functions (sqrt, trig, exp, …) strip the physical dimension
    return ""


def _function_unit_multi(func_name: str, first_arg_unit: str) -> str:
    """Infer the unit produced by a multi-argument mathematical function call."""
    lower = func_name.lower()
    # atan2 returns an angle in radians — treated as dimensionless here
    if lower == "atan2":
        return ""
    # pow / pwr / pwrs: unit exponentiation is not well-defined in general
    if lower in ("pow", "pwr", "pwrs"):
        return ""
    # min / max / limit propagate the first argument's physical unit
    if lower in ("min", "max", "limit"):
        return first_arg_unit
    return ""


class ExpressionEvaluator:
    """Stateless service that walks an expression AST and returns an
    :class:`~viewer.expression.Expression`.

    Usage example::

        parser    = ExpressionParser()
        evaluator = ExpressionEvaluator()

        tree    = parser.parse("V(R1) / I(R1)")
        context = {"V(R1)": expr_vr1, "I(R1)": expr_ir1}
        result  = evaluator.evaluate(tree, context, name="Z(R1)")
        # result.unit == "Ω"
    """

    def evaluate(self, node: ExpressionNode, context: dict[str, Expression], name: str | None = None, source: str | None = None) -> Expression:
        """Evaluate *node* against *context* and return an :class:`~viewer.expression.Expression`.

        :param node:    Root AST node returned by :class:`~viewer.expression_parser.ExpressionParser`.
        :param context: Mapping from variable name to :class:`~viewer.expression.Expression`.
                        Lookup is case-insensitive.
        :param name:    Display name for the resulting ``Expression``.  Defaults to
                        a string reconstruction of the AST.
        :param source:  Original source expression string stored on the result for
                        display / debugging.  Defaults to *name*.
        :raises ValueError: If a variable is not found in *context* or an unknown
                            function is encountered.
        """
        # evaluate the AST and propagate units
        data, unit = self._eval(node, context)
        # use the provided name or reconstruct one from the AST
        expr_name = name if name is not None else str(node)
        # exit
        return Expression(expr_name, data, unit, source=source if source is not None else expr_name)

    def _eval(self, node: ExpressionNode, context: dict[str, Expression]) -> tuple[np.ndarray, str]:
        """Recursively evaluate *node*, returning ``(data_array, unit_string)``."""
        # number literal evaluates to a scalar array with empty unit
        if isinstance(node, NumberNode):
            return np.array(node.value), ""
        # built-in constants shadow any simulation variable with the same name (case-insensitive)
        if isinstance(node, VariableRefNode):
            # check built-in constants first (pi, mho)
            entry = _CONSTANTS.get(node.name.lower())
            if entry is not None:
                # entry is a (value, unit) pair, e.g. (3.14159..., "") for pi or (1.0, "S") for mho
                const_value, const_unit = entry
                # exit
                return np.array(const_value), const_unit
            try:
                # lookup variable in context
                var = self._lookup(node.name, context)
                # exit
                return var.data, var.unit
            except ValueError:
                # not exists in context
                pass
            # two-node differential form: FUNC(a, b) = FUNC(a) - FUNC(b) per SPICE convention;
            # node "0" is the SPICE ground reference and is never stored as a variable (it is
            # always 0 V), so it is treated as absent and substituted with zeros.
            m = _TWO_ARG_RE.match(node.name)
            if m:
                func = m.group(1)
                arg_a = m.group(2).strip()
                arg_b = m.group(3).strip()
                va = None if arg_a == "0" else self._lookup(f"{func}({arg_a})", context)
                vb = None if arg_b == "0" else self._lookup(f"{func}({arg_b})", context)
                # use whichever non-ground node is available as the reference for shape and unit
                ref = va if va is not None else vb
                if ref is not None:
                    da = va.data if va is not None else np.zeros_like(ref.data)
                    db = vb.data if vb is not None else np.zeros_like(ref.data)
                    return da - db, ref.unit
            raise ValueError(f"undefined variable: {node.name!r}")
        # function call: delegate to _eval_function
        if isinstance(node, FunctionCallNode):
            return self._eval_function(node, context)
        # binary operation: evaluate both sides, apply the operator, and propagate units
        if isinstance(node, BinaryOperatorNode):
            left_data, left_unit = self._eval(node.left, context)
            right_data, right_unit = self._eval(node.right, context)
            data = self._apply_binary_op(node.op, left_data, right_data)
            unit = _propagate_binary_unit(left_unit, node.op, right_unit)
            return data, unit
        # unary operation: evaluate the operand and apply the unary operator
        if isinstance(node, UnaryOperatorNode):
            data, unit = self._eval(node.operand, context)
            if node.op == UnaryOperator.NEG:
                return -data, unit
            raise ValueError(f"unsupported unary operator: {node.op}")
        raise ValueError(f"unknown AST node type: {type(node).__name__}")

    def _eval_function(self, node: FunctionCallNode, context: dict[str, Expression]) -> tuple[np.ndarray, str]:
        lower = node.name.lower()
        # single-argument functions
        func1 = _FUNCTION_IMPLS.get(lower)
        if func1 is not None:
            if len(node.args) != 1:
                raise ValueError(f"function {node.name!r} expects exactly 1 argument, got {len(node.args)}")
            arg_data, arg_unit = self._eval(node.args[0], context)
            result_data = func1(arg_data)
            result_unit = _function_unit(node.name, arg_unit)
            return result_data, result_unit
        # two-argument functions
        func2 = _FUNCTION_IMPLS_2.get(lower)
        if func2 is not None:
            if len(node.args) != 2:
                raise ValueError(f"function {node.name!r} expects exactly 2 arguments, got {len(node.args)}")
            a_data, a_unit = self._eval(node.args[0], context)
            b_data, _ = self._eval(node.args[1], context)
            result_data = func2(a_data, b_data)
            result_unit = _function_unit_multi(node.name, a_unit)
            return result_data, result_unit
        # three-argument functions
        func3 = _FUNCTION_IMPLS_3.get(lower)
        if func3 is not None:
            if len(node.args) != 3:
                raise ValueError(f"function {node.name!r} expects exactly 3 arguments, got {len(node.args)}")
            a_data, a_unit = self._eval(node.args[0], context)
            b_data, _ = self._eval(node.args[1], context)
            c_data, _ = self._eval(node.args[2], context)
            result_data = func3(a_data, b_data, c_data)
            result_unit = _function_unit_multi(node.name, a_unit)
            return result_data, result_unit
        # raise for unknown functions
        raise ValueError(f"unknown function: {node.name!r}")

    @staticmethod
    def _apply_binary_op(op: BinaryOperator, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if op == BinaryOperator.ADD:
            return left + right
        if op == BinaryOperator.SUB:
            return left - right
        if op == BinaryOperator.MUL:
            return left * right
        if op == BinaryOperator.DIV:
            return left / right
        if op == BinaryOperator.POW:
            return left ** right
        raise ValueError(f"unsupported binary operator: {op}")

    @staticmethod
    def _lookup(name: str, context: dict[str, Expression]) -> Expression:
        """Look up *name* in *context* with case-insensitive fallback."""
        # try an exact match first
        var = context.get(name)
        if var is not None:
            return var
        # fall back to a case-insensitive scan
        lower = name.lower()
        for key, v in context.items():
            if key.lower() == lower:
                return v
        raise ValueError(f"undefined variable: {name!r}")
