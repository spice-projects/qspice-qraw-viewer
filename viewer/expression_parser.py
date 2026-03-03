"""Recursive-descent parser for SPICE/QSPICE waveform expressions."""
from __future__ import annotations

import re
from typing import NamedTuple

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

# Known mathematical functions that accept expressions as arguments.
# Everything else of the form  IDENT(...)  is treated as a SPICE variable
# reference whose full text (e.g. "V(out)", "I(R1,0)") is the lookup key.
_MATH_FUNCTIONS: frozenset[str] = frozenset({
    "abs", "sqrt", "log", "log10", "ln",
    "db", "dB",
    "real", "imag", "angle", "mag",
    "sin", "cos", "tan",
    "exp",
})

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class _Token(NamedTuple):
    type: str
    value: str


_TOKEN_RE = re.compile(
    r"(?P<NUMBER>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_\[\]]*)"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<PLUS>\+)"
    r"|(?P<MINUS>-)"
    r"|(?P<STAR>\*)"
    r"|(?P<SLASH>/)"
    r"|(?P<CARET>\^)"
    r"|(?P<SPACE>\s+)"
)


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise ValueError(f"unexpected character {text[pos]!r} at position {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "SPACE":
            continue
        tokens.append(_Token(kind, m.group()))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ExpressionParser:
    """Stateless recursive-descent parser for waveform expressions.

    Supported syntax::

        expr        ::= additive
        additive    ::= multiplicative (( '+' | '-' ) multiplicative)*
        multiplicative ::= unary (( '*' | '/' ) unary)*
        unary       ::= '-' unary | power
        power       ::= primary ( '^' unary )*
        primary     ::= NUMBER
                      | IDENT '(' expr_arglist ')'   -- math function
                      | IDENT '(' raw_arglist  ')'   -- SPICE variable ref
                      | IDENT                        -- bare variable ref
                      | '(' expr ')'

    SPICE variable references such as ``V(R1)``, ``V(out,0)`` or ``I(C1)``
    are returned as :class:`~viewer.expression_node.VariableRefNode` instances
    whose ``name`` is the full reconstructed string used as the context lookup
    key.  Mathematical function calls such as ``db(x)`` or ``abs(x)`` are
    returned as :class:`~viewer.expression_node.FunctionCallNode` instances
    with their arguments parsed as full sub-expressions.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ExprNode:
        """Parse *text* into an expression tree.

        :param text: Raw expression string, e.g. ``"V(R1,0) / I(R1)"``
        :returns: Root :class:`~viewer.expression_node.ExprNode` of the AST.
        :raises ValueError: If the expression contains a syntax error.
        """
        self._tokens: list[_Token] = _tokenize(text)
        self._pos: int = 0
        node = self._parse_additive()
        if self._pos < len(self._tokens):
            raise ValueError(f"unexpected token {self._tokens[self._pos].value!r}")
        return node

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def _peek(self) -> _Token | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_type: str | None = None) -> _Token:
        if self._pos >= len(self._tokens):
            raise ValueError("unexpected end of expression")
        tok = self._tokens[self._pos]
        if expected_type is not None and tok.type != expected_type:
            raise ValueError(f"expected {expected_type!r}, got {tok.value!r}")
        self._pos += 1
        return tok

    def _parse_additive(self) -> ExprNode:
        node = self._parse_multiplicative()
        while (tok := self._peek()) and tok.type in ("PLUS", "MINUS"):
            self._consume()
            right = self._parse_multiplicative()
            op = BinaryOp.ADD if tok.value == "+" else BinaryOp.SUB
            node = BinaryOpNode(node, op, right)
        return node

    def _parse_multiplicative(self) -> ExprNode:
        node = self._parse_unary()
        while (tok := self._peek()) and tok.type in ("STAR", "SLASH"):
            self._consume()
            right = self._parse_unary()
            op = BinaryOp.MUL if tok.value == "*" else BinaryOp.DIV
            node = BinaryOpNode(node, op, right)
        return node

    def _parse_unary(self) -> ExprNode:
        if (tok := self._peek()) and tok.type == "MINUS":
            self._consume()
            operand = self._parse_power()
            return UnaryOpNode(UnaryOp.NEG, operand)
        return self._parse_power()

    def _parse_power(self) -> ExprNode:
        base = self._parse_primary()
        if (tok := self._peek()) and tok.type == "CARET":
            self._consume()
            exp = self._parse_unary()  # right-associative
            return BinaryOpNode(base, BinaryOp.POW, exp)
        return base

    def _parse_primary(self) -> ExprNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of expression")

        # Numeric literal
        if tok.type == "NUMBER":
            self._consume()
            return NumberNode(float(tok.value))

        # Identifier — either a function call or a bare variable reference
        if tok.type == "IDENT":
            self._consume()
            name = tok.value
            if self._peek() and self._peek().type == "LPAREN":
                self._consume()  # consume '('
                lower_name = name.lower()
                if lower_name in _MATH_FUNCTIONS:
                    args = self._parse_expr_arglist()
                    self._consume("RPAREN")
                    return FunctionCallNode(name, args)
                else:
                    raw_args = self._parse_raw_arglist()
                    self._consume("RPAREN")
                    args_str = ", ".join(raw_args)
                    return VariableRefNode(f"{name}({args_str})")
            return VariableRefNode(name)

        # Parenthesised sub-expression
        if tok.type == "LPAREN":
            self._consume()
            node = self._parse_additive()
            self._consume("RPAREN")
            return node

        raise ValueError(f"unexpected token {tok.value!r}")

    # ------------------------------------------------------------------
    # Argument list helpers
    # ------------------------------------------------------------------

    def _parse_expr_arglist(self) -> list[ExprNode]:
        """Parse a comma-separated list of full sub-expressions."""
        args: list[ExprNode] = []
        if self._peek() and self._peek().type != "RPAREN":
            args.append(self._parse_additive())
            while self._peek() and self._peek().type == "COMMA":
                self._consume()
                args.append(self._parse_additive())
        return args

    def _parse_raw_arglist(self) -> list[str]:
        """Collect raw (non-expression) argument strings for SPICE probe refs.

        Inside ``V(...)`` or ``I(...)`` the arguments are node names such as
        ``"R1"``, ``"0"``, ``"outd[3]"`` — not arithmetic sub-expressions.
        This method collects tokens verbatim until the matching ``)``.
        """
        raw_args: list[str] = []
        current: list[str] = []
        while (tok := self._peek()) and tok.type != "RPAREN":
            if tok.type == "COMMA":
                raw_args.append("".join(current).strip())
                current = []
                self._consume()
            else:
                current.append(tok.value)
                self._consume()
        if current:
            raw_args.append("".join(current).strip())
        return raw_args
