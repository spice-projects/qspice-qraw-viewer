import logging

import numpy as np

from .expression import Expression
from .qspice_language.evaluator import QspiceEvaluator
from .qspice_language.nodes import BinaryOperationNode, BinaryOperator, ExpressionNode, FunctionCallNode, FunctionDefinitionNode, IdentifierNode, NumberNode, TernaryOperationNode, UnaryOperationNode
from .qspice_language.parser import QspiceParser

logger = logging.getLogger(__name__)


class ExpressionManager:

    _CONSTANT_UNITS: dict[str, str] = {
        "e": "",
        "f": "",
        "g": "",
        "k": "",
        "m": "",
        "meg": "",
        "mho": "S",
        "n": "",
        "p": "",
        "pi": "",
            "s": "s",
        "t": "",
        "u": "",
    }

    def __init__(self, expressions: list[Expression], function_definitions: list[str] | None = None):
        # create expression context; keys are lowercased so that evaluate() lookups always match
        self._context: dict[str, Expression] = {expression.name.lower(): expression for expression in expressions}
        # initialize the qspice language parser and evaluator for expression data
        self._parser = QspiceParser()
        self._evaluator = QspiceEvaluator()
        # parse user-defined .func definitions; keys are lowercased for case-insensitive lookup
        self._functions: dict[str, FunctionDefinitionNode] = {}
        for func_text in (function_definitions or []):
            try:
                definition = self._parser.parse_function_definition(func_text)
                self._functions[definition.name.casefold()] = definition
            except ValueError as e:
                logger.warning("Failed to parse .func definition %r: %s", func_text, e)

    @property
    def expressions(self) -> list[Expression]:
        # do not show calculated expressions in the list of expressions
        return list(self._context.values())

    def evaluate(self, expression: str, name: str | None = None) -> Expression | None:
        # context key
        key = (name or expression).lower()
        # check expression has been evaluated before
        result = self._context.get(key, None)
        if result is None:
            try:
                # parse the expression string into a qspice AST
                ast = self._parser.parse_expression(expression)
                # build qspice variable context from cached expressions
                data_context = {context_key: context_expression.data for context_key, context_expression in self._context.items()}
                # evaluate qspice AST to get computed numeric data
                evaluated_data = self._evaluator.evaluate(ast, data_context, self._functions)
                # build unit lookup context from cached expressions
                unit_context = {context_key: context_expression.unit for context_key, context_expression in self._context.items()}
                # infer propagated unit from the qspice AST
                inferred_unit = self._infer_unit(ast, unit_context)
                # normalize expression name
                result_name = name if name is not None else self._format_expression(ast)
                # build the final expression using qspice data and qspice-based unit propagation
                result = Expression(result_name, np.asarray(evaluated_data), inferred_unit, source="expression manager")
                # update context with the evaluated expression for future reference
                self._context[key] = result
            except ValueError as e:
                # log information
                logger.warning("Failed to evaluate expression %r: %s", expression, e)
        # exit
        return result

    def _infer_unit(self, node: ExpressionNode, unit_context: dict[str, str]) -> str:
        # numeric literals are dimensionless
        if isinstance(node, NumberNode):
            return ""
        # resolve identifier unit from context or built-in constants
        if isinstance(node, IdentifierNode):
            return self._unit_for_identifier(node.name, unit_context)
        # unary operators preserve the operand unit
        if isinstance(node, UnaryOperationNode):
            return self._infer_unit(node.operand, unit_context)
        # infer unit for binary operators from both sides
        if isinstance(node, BinaryOperationNode):
            left_unit = self._infer_unit(node.left, unit_context)
            right_unit = self._infer_unit(node.right, unit_context)
            return self._propagate_binary_unit(left_unit, node.operator, right_unit)
        # infer unit for function calls from the first argument by convention
        if isinstance(node, FunctionCallNode):
            function_name = node.name.casefold()
            # resolve probe units using probe references
            if function_name in ("v", "i", "id"):
                return self._unit_for_probe(node, unit_context)
            # delegate unit inference to user-defined function body
            definition = self._functions.get(function_name)
            if definition is not None:
                return self._infer_user_function_unit(definition, node, unit_context)
            # dimensionless fallback for nullary calls
            if len(node.args) == 0:
                return ""
            # infer first argument unit
            first_arg_unit = self._infer_unit(node.args[0], unit_context)
            # infer unit for unary function calls
            if len(node.args) == 1:
                return self._function_unit(function_name, first_arg_unit)
            # infer unit for multi-argument function calls
            return self._function_unit_multi(function_name, first_arg_unit)
        # infer a common unit for ternary branches
        if isinstance(node, TernaryOperationNode):
            true_unit = self._infer_unit(node.if_true, unit_context)
            false_unit = self._infer_unit(node.if_false, unit_context)
            return true_unit if true_unit == false_unit else ""
        # default to dimensionless for unsupported nodes
        return ""

    def _format_expression(self, node: ExpressionNode) -> str:
        # format numeric literals
        if isinstance(node, NumberNode):
            return node.text
        # format identifiers
        if isinstance(node, IdentifierNode):
            return node.name
        # format unary expressions
        if isinstance(node, UnaryOperationNode):
            return node.operator.value + self._format_expression(node.operand)
        # format binary expressions with explicit grouping
        if isinstance(node, BinaryOperationNode):
            return "(" + self._format_expression(node.left) + node.operator.value + self._format_expression(node.right) + ")"
        # format function calls
        if isinstance(node, FunctionCallNode):
            return node.name + "(" + ",".join(self._format_expression(arg) for arg in node.args) + ")"
        # format ternary expressions with explicit grouping
        if isinstance(node, TernaryOperationNode):
            return "(" + self._format_expression(node.condition) + "?" + self._format_expression(node.if_true) + ":" + self._format_expression(node.if_false) + ")"
        # fallback for unknown nodes
        return ""

    def _infer_user_function_unit(self, definition: FunctionDefinitionNode, call: FunctionCallNode, unit_context: dict[str, str]) -> str:
        # build a unit context that maps each parameter name to the unit of the corresponding argument
        local_unit_context = dict(unit_context)
        for param, arg in zip(definition.params, call.args):
            local_unit_context[param.casefold()] = self._infer_unit(arg, unit_context)
        # infer unit by walking the function body with the local context
        return self._infer_unit(definition.body, local_unit_context)

    def _unit_for_identifier(self, name: str, unit_context: dict[str, str]) -> str:
        # normalize the identifier lookup key
        key = name.casefold()
        # resolve context expression unit
        unit = unit_context.get(key)
        if unit is not None:
            return unit
        # resolve built-in constant unit
        constant_unit = self._CONSTANT_UNITS.get(key)
        if constant_unit is not None:
            return constant_unit
        # fallback to dimensionless
        return ""

    def _unit_for_probe(self, probe: FunctionCallNode, unit_context: dict[str, str]) -> str:
        # reconstruct the probe key used by the context
        probe_key = self._probe_key(probe).casefold()
        # resolve a directly stored probe unit
        direct_unit = unit_context.get(probe_key)
        if direct_unit is not None:
            return direct_unit
        # infer known probe family units
        lower_name = probe.name.casefold()
        if lower_name == "v":
            return "V"
        if lower_name in ("i", "id"):
            return "A"
        return ""

    @staticmethod
    def _probe_key(probe: FunctionCallNode) -> str:
        # reconstruct probe text with normalized comma spacing
        args = ", ".join(ExpressionManager._probe_arg_text(arg) for arg in probe.args)
        return f"{probe.name}({args})"

    @staticmethod
    def _probe_arg_text(arg: ExpressionNode) -> str:
        # serialize identifier arguments directly
        if isinstance(arg, IdentifierNode):
            return arg.name
        # serialize numeric arguments directly
        if isinstance(arg, NumberNode):
            return arg.text
        # unsupported argument forms are rendered empty
        return ""

    @staticmethod
    def _propagate_binary_unit(left_unit: str, operator: BinaryOperator, right_unit: str) -> str:
        # addition and subtraction require matching units
        if operator in (BinaryOperator.ADD, BinaryOperator.SUB):
            return left_unit if left_unit == right_unit else ""
        # multiplication handles common electrical identities
        if operator == BinaryOperator.MUL:
            if {left_unit, right_unit} == {"V", "A"}:
                return "W"
            if {left_unit, right_unit} == {"S", "V"}:
                return "A"
            if left_unit == "":
                return right_unit
            return left_unit
        # division handles common reciprocal electrical identities
        if operator == BinaryOperator.DIV:
            if left_unit == right_unit:
                return ""
            if left_unit == "V" and right_unit == "A":
                return "Ω"
            if left_unit == "A" and right_unit == "V":
                return "S"
            if right_unit == "":
                return left_unit
            if left_unit == "":
                if right_unit == "S":
                    return "Ω"
                if right_unit == "Ω":
                    return "S"
                if right_unit == "s":
                    return "Hz"
                if right_unit == "Hz":
                    return "s"
            return ""
        # all other operators are treated as dimensionless
        return ""

    @staticmethod
    def _function_unit(function_name: str, arg_unit: str) -> str:
        # db always returns decibels
        if function_name == "db":
            return "dB"
        # angle aliases return degrees
        if function_name in ("angle", "ph", "phase"):
            return "°"
        # these functions preserve their argument unit
        if function_name in ("abs", "real", "imag", "mag", "conj", "uramp", "round", "floor", "ceil", "int"):
            return arg_unit
        # all remaining unary functions are dimensionless
        return ""

    @staticmethod
    def _function_unit_multi(function_name: str, first_arg_unit: str) -> str:
        # min, max, and limit preserve the first argument unit
        if function_name in ("min", "max", "limit"):
            return first_arg_unit
        # all remaining multi-argument functions are dimensionless
        return ""
