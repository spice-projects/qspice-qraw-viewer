"""AST node types produced by :class:`~viewer.expression_parser.ExpressionParser`."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class BinaryOp(Enum):
    """Binary arithmetic operators supported in expressions."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"


class UnaryOp(Enum):
    """Unary operators supported in expressions."""

    NEG = "-"


@dataclass
class NumberNode:
    """A numeric literal constant."""

    value: float


@dataclass
class VariableRefNode:
    """A reference to a simulation variable by its full name.

    The ``name`` matches the key used to look up the variable in the
    evaluation context (e.g. ``"V(R1)"``, ``"I(R1,0)"``, ``"Vout"``).
    """

    name: str


@dataclass
class FunctionCallNode:
    """A call to a known mathematical function (e.g. ``db``, ``abs``, ``sqrt``)."""

    name: str
    args: list[ExprNode] = field(default_factory=list)


@dataclass
class BinaryOpNode:
    """A binary arithmetic operation between two sub-expressions."""

    left: ExprNode
    op: BinaryOp
    right: ExprNode


@dataclass
class UnaryOpNode:
    """A unary operation applied to a single sub-expression."""

    op: UnaryOp
    operand: ExprNode


# union type for all AST nodes
ExprNode = Union[NumberNode, VariableRefNode, FunctionCallNode, BinaryOpNode, UnaryOpNode]
