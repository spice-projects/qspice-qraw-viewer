"""Recursive-descent parser for SPICE/QSPICE waveform expressions."""
from __future__ import annotations

import re
from typing import NamedTuple

from .expression_node import BinaryOp, BinaryOpNode, ExprNode, FunctionCallNode, NumberNode, UnaryOp, UnaryOpNode, VariableRefNode

# known mathematical functions that accept expressions as arguments;
# everything else of the form IDENT(...) is treated as a SPICE variable
# reference whose full text (e.g. "V(out)", "I(R1,0)") is the lookup key
_MATH_FUNCTIONS: frozenset[str] = frozenset({"abs", "sqrt", "log", "log10", "ln", "db", "dB", "real", "imag", "angle", "mag", "sin", "cos", "tan", "exp"})


class _Token(NamedTuple):
    type: str
    value: str


_TOKEN_RE = re.compile(r"(?P<NUMBER>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|(?P<IDENT>[A-Za-z_][A-Za-z0-9_\[\]]*)|(?P<LPAREN>\()|(?P<RPAREN>\))|(?P<COMMA>,)|(?P<PLUS>\+)|(?P<MINUS>-)|(?P<STAR>\*)|(?P<SLASH>/)|(?P<CARET>\^)|(?P<SPACE>\s+)")


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        # raise if no token matched at the current position
        if m is None:
            raise ValueError(f"unexpected character {text[pos]!r} at position {pos}")
        pos = m.end()
        kind = m.lastgroup
        # skip whitespace tokens
        if kind == "SPACE":
            continue
        tokens.append(_Token(kind, m.group()))
    # exit
    return tokens


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

    def parse(self, text: str) -> ExprNode:
        """Parse *text* into an expression tree.

        :param text: Raw expression string, e.g. ``"V(R1,0) / I(R1)"``
        :returns: Root :class:`~viewer.expression_node.ExprNode` of the AST.
        :raises ValueError: If the expression contains a syntax error.
        """
        # tokenize the input and reset the position cursor
        self._tokens: list[_Token] = _tokenize(text)
        self._pos: int = 0
        # parse the full expression starting at the additive level
        node = self._parse_additive()
        # any remaining tokens indicate a syntax error
        if self._pos < len(self._tokens):
            raise ValueError(f"unexpected token {self._tokens[self._pos].value!r}")
        # exit
        return node

    def _peek(self) -> _Token | None:
        # return the token at the current position without consuming it, or None if at end
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_type: str | None = None) -> _Token:
        # raise if there are no more tokens to consume
        if self._pos >= len(self._tokens):
            raise ValueError("unexpected end of expression")
        tok = self._tokens[self._pos]
        # raise if the current token type does not match the expected type
        if expected_type is not None and tok.type != expected_type:
            raise ValueError(f"expected {expected_type!r}, got {tok.value!r}")
        self._pos += 1
        # exit
        return tok

    def _parse_additive(self) -> ExprNode:
        # parse the left-hand side of a potential addition or subtraction
        node = self._parse_multiplicative()
        # consume any + or - operators and fold the right-hand side into a BinaryOpNode
        while (tok := self._peek()) and tok.type in ("PLUS", "MINUS"):
            self._consume()
            right = self._parse_multiplicative()
            op = BinaryOp.ADD if tok.value == "+" else BinaryOp.SUB
            node = BinaryOpNode(node, op, right)
        # exit
        return node

    def _parse_multiplicative(self) -> ExprNode:
        # parse the left-hand side of a potential multiplication or division
        node = self._parse_unary()
        # consume any * or / operators and fold the right-hand side into a BinaryOpNode
        while (tok := self._peek()) and tok.type in ("STAR", "SLASH"):
            self._consume()
            right = self._parse_unary()
            op = BinaryOp.MUL if tok.value == "*" else BinaryOp.DIV
            node = BinaryOpNode(node, op, right)
        # exit
        return node

    def _parse_unary(self) -> ExprNode:
        # handle unary minus by wrapping the operand in a UnaryOpNode
        if (tok := self._peek()) and tok.type == "MINUS":
            self._consume()
            operand = self._parse_power()
            return UnaryOpNode(UnaryOp.NEG, operand)
        # exit
        return self._parse_power()

    def _parse_power(self) -> ExprNode:
        # parse the base of a potential power expression
        base = self._parse_primary()
        # if a ^ follows, parse the exponent right-associatively
        if (tok := self._peek()) and tok.type == "CARET":
            self._consume()
            # right-associative: exponent is parsed at unary precedence
            exp = self._parse_unary()
            return BinaryOpNode(base, BinaryOp.POW, exp)
        # exit
        return base

    def _parse_primary(self) -> ExprNode:
        # peek at the next token to determine what kind of primary to parse
        tok = self._peek()
        # raise if the expression is empty or unexpectedly ends here
        if tok is None:
            raise ValueError("unexpected end of expression")
        # numeric literal
        if tok.type == "NUMBER":
            self._consume()
            value = float(tok.value)
            # implicit multiplication: NUMBER directly followed by a bare IDENT — handles
            # SPICE unit suffixes such as 'mho' in expressions like '1mho*V(out,0)'
            next_tok = self._peek()
            if next_tok and next_tok.type == "IDENT":
                lookahead_pos = self._pos + 1
                after_ident = self._tokens[lookahead_pos] if lookahead_pos < len(self._tokens) else None
                # only treat as implicit multiplication when the IDENT is not a function call (not followed by '(')
                if after_ident is None or after_ident.type != "LPAREN":
                    self._consume()
                    return BinaryOpNode(NumberNode(value), BinaryOp.MUL, VariableRefNode(next_tok.value))
            return NumberNode(value)
        # identifier — either a function call or a bare variable reference
        if tok.type == "IDENT":
            self._consume()
            name = tok.value
            if self._peek() and self._peek().type == "LPAREN":
                # consume '('
                self._consume()
                lower_name = name.lower()
                # known math functions get their arguments parsed as sub-expressions
                if lower_name in _MATH_FUNCTIONS:
                    args = self._parse_expr_arglist()
                    self._consume("RPAREN")
                    return FunctionCallNode(name, args)
                # all other IDENT(...) forms are SPICE variable references
                raw_args = self._parse_raw_arglist()
                self._consume("RPAREN")
                args_str = ", ".join(raw_args)
                return VariableRefNode(f"{name}({args_str})")
            return VariableRefNode(name)
        # parenthesised sub-expression
        if tok.type == "LPAREN":
            self._consume()
            node = self._parse_additive()
            self._consume("RPAREN")
            return node
        raise ValueError(f"unexpected token {tok.value!r}")

    def _parse_expr_arglist(self) -> list[ExprNode]:
        """Parse a comma-separated list of full sub-expressions."""
        args: list[ExprNode] = []
        # parse the first argument if the list is non-empty
        if self._peek() and self._peek().type != "RPAREN":
            args.append(self._parse_additive())
            # consume any additional comma-separated arguments
            while self._peek() and self._peek().type == "COMMA":
                self._consume()
                args.append(self._parse_additive())
        # exit
        return args

    def _parse_raw_arglist(self) -> list[str]:
        """Collect raw (non-expression) argument strings for SPICE probe refs.

        Inside ``V(...)`` or ``I(...)`` the arguments are node names such as
        ``"R1"``, ``"0"``, ``"outd[3]"`` — not arithmetic sub-expressions.
        This method collects tokens verbatim until the matching ``)``.
        """
        raw_args: list[str] = []
        current: list[str] = []
        # collect tokens until the closing parenthesis
        while (tok := self._peek()) and tok.type != "RPAREN":
            if tok.type == "COMMA":
                raw_args.append("".join(current).strip())
                current = []
                self._consume()
            else:
                current.append(tok.value)
                self._consume()
        # flush any remaining tokens into the last argument
        if current:
            raw_args.append("".join(current).strip())
        # exit
        return raw_args
