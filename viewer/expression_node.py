"""AST node types produced by :class:`~viewer.expression_parser.ExpressionParser`."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class BinaryOperator(Enum):

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"


class UnaryOperator(Enum):

    NEG = "-"


@dataclass
class NumberNode:

    value: float


@dataclass
class VariableRefNode:

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class FunctionCallNode:

    name: str
    args: list[ExpressionNode] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}({",".join(str(a) for a in self.args)})"


@dataclass
class BinaryOperatorNode:

    left: ExpressionNode
    op: BinaryOperator
    right: ExpressionNode

    def __str__(self) -> str:
        return f"({self.left}{self.op.value}{self.right})"


@dataclass
class UnaryOperatorNode:

    op: UnaryOperator
    operand: ExpressionNode

    def __str__(self) -> str:
        return f"{self.op.value}{self.operand}"


# union type for all AST nodes
ExpressionNode = Union[NumberNode, VariableRefNode, FunctionCallNode, BinaryOperatorNode, UnaryOperatorNode]
