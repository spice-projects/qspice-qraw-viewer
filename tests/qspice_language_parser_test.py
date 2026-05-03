from unittest import TestCase

from viewer.qspice_language.nodes import BinaryOperationNode
from viewer.qspice_language.nodes import BinaryOperator
from viewer.qspice_language.nodes import FunctionCallNode
from viewer.qspice_language.nodes import FunctionDefinitionNode
from viewer.qspice_language.nodes import IdentifierNode
from viewer.qspice_language.nodes import NumberNode
from viewer.qspice_language.nodes import StepSelectorNode
from viewer.qspice_language.nodes import TernaryOperationNode
from viewer.qspice_language.nodes import UnaryOperationNode
from viewer.qspice_language.nodes import UnaryOperator
from viewer.qspice_language.parser import QspiceParser


class TestQspiceParser(TestCase):

    def test_parse_number_literal(self):
        # arrange
        parser = QspiceParser()
        # act — pure number without suffix
        expression = parser.parse_expression("1e-05")
        # assert
        self.assertIsInstance(expression, NumberNode)
        self.assertEqual(expression.text, "1e-05")

    def test_parse_number_with_implicit_multiplication_suffix(self):
        # arrange — SPICE suffixes like 'meg' are now tokenized separately for implicit multiplication
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("1e-05meg")
        # assert — parses as NUMBER * IDENTIFIER (implicit multiplication)
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.MUL)
        self.assertIsInstance(expression.left, NumberNode)
        self.assertEqual(expression.left.text, "1e-05")
        self.assertIsInstance(expression.right, IdentifierNode)
        self.assertEqual(expression.right.name, "meg")

    def test_parse_identifier(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("gain")
        # assert
        self.assertIsInstance(expression, IdentifierNode)
        self.assertEqual(expression.name, "gain")

    def test_parse_function_call(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("max(a, 2)")
        # assert
        self.assertIsInstance(expression, FunctionCallNode)
        self.assertEqual(expression.name, "max")
        self.assertEqual(len(expression.args), 2)

    def test_parse_probe_selector_as_step_selector_node(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("V(INOISE)@1")
        # assert — @N is now a StepSelectorNode, not a folded string identifier
        self.assertIsInstance(expression, StepSelectorNode)
        self.assertEqual(expression.step_index, 1)
        self.assertIsInstance(expression.base, FunctionCallNode)
        self.assertEqual(expression.base.name, "V")

    def test_parse_precedence_mul_before_add(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("a + b * c")
        # assert
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.ADD)
        self.assertIsInstance(expression.right, BinaryOperationNode)
        self.assertEqual(expression.right.operator, BinaryOperator.MUL)

    def test_parse_precedence_power_before_multiplication(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("a * b ** 2")
        # assert
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.MUL)
        self.assertIsInstance(expression.right, BinaryOperationNode)
        self.assertEqual(expression.right.operator, BinaryOperator.POW)

    def test_parse_caret_as_power_operator(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("a ^ 2")
        # assert
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.POW)

    def test_parse_logical_and_relational_precedence(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("a < b && c < d")
        # assert
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.LOGICAL_AND)
        self.assertIsInstance(expression.left, BinaryOperationNode)
        self.assertEqual(expression.left.operator, BinaryOperator.LESS)

    def test_parse_ternary_expression(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("x > 0 ? x : -x")
        # assert
        self.assertIsInstance(expression, TernaryOperationNode)
        self.assertIsInstance(expression.condition, BinaryOperationNode)
        self.assertIsInstance(expression.if_false, UnaryOperationNode)
        self.assertEqual(expression.if_false.operator, UnaryOperator.NEG)

    def test_parse_function_definition(self):
        # arrange
        parser = QspiceParser()
        # act
        definition = parser.parse_function_definition(".func gain(x, y) {x / y}")
        # assert
        self.assertIsInstance(definition, FunctionDefinitionNode)
        self.assertEqual(definition.name, "gain")
        self.assertEqual(definition.params, ("x", "y"))
        self.assertIsInstance(definition.body, BinaryOperationNode)
        self.assertEqual(definition.body.operator, BinaryOperator.DIV)

    def test_parse_function_definition_with_probe_selectors(self):
        # arrange
        parser = QspiceParser()
        # act
        definition = parser.parse_function_definition(".func NFDB(){20*LOG10(V(INOISE)@1/V(INOISE)@2)}")
        # assert — body must contain StepSelectorNodes for the two step references
        self.assertIsInstance(definition, FunctionDefinitionNode)
        self.assertEqual(definition.name, "NFDB")
        self.assertEqual(definition.params, ())

    def test_parse_nested_ternary_is_right_associative(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("a ? b : c ? d : e")
        # assert
        self.assertIsInstance(expression, TernaryOperationNode)
        self.assertIsInstance(expression.if_false, TernaryOperationNode)

    def test_parse_missing_closing_paren_raises_error(self):
        # arrange
        parser = QspiceParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse_expression("max(a, 2")

    # ------------------------------------------------------------------ #
    # Step selectors (@N)                                                  #
    # ------------------------------------------------------------------ #

    def test_parse_step_selector_on_probe_produces_step_selector_node(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("V(INOISE)@2")
        # assert
        self.assertIsInstance(expression, StepSelectorNode)
        self.assertEqual(expression.step_index, 2)
        self.assertIsInstance(expression.base, FunctionCallNode)
        self.assertEqual(expression.base.name, "V")

    def test_parse_step_selector_on_identifier_produces_step_selector_node(self):
        # arrange
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("gain@3")
        # assert
        self.assertIsInstance(expression, StepSelectorNode)
        self.assertEqual(expression.step_index, 3)
        self.assertIsInstance(expression.base, IdentifierNode)
        self.assertEqual(expression.base.name, "gain")

    def test_parse_step_selector_index_one(self):
        # arrange — @1 is the minimum valid step index
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("V(a)@1")
        # assert
        self.assertIsInstance(expression, StepSelectorNode)
        self.assertEqual(expression.step_index, 1)

    def test_parse_step_selector_zero_raises_error(self):
        # arrange — @0 is invalid: step indices are 1-based
        parser = QspiceParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse_expression("V(a)@0")

    def test_parse_step_selector_in_expression_tree(self):
        # arrange — V(INOISE)@1/V(INOISE)@2 should produce a division of two selectors
        parser = QspiceParser()
        # act
        expression = parser.parse_expression("V(INOISE)@1/V(INOISE)@2")
        # assert
        self.assertIsInstance(expression, BinaryOperationNode)
        self.assertEqual(expression.operator, BinaryOperator.DIV)
        self.assertIsInstance(expression.left, StepSelectorNode)
        self.assertEqual(expression.left.step_index, 1)
        self.assertIsInstance(expression.right, StepSelectorNode)
        self.assertEqual(expression.right.step_index, 2)

    def test_parse_nfdb_func_definition_body_contains_step_selectors(self):
        # arrange — real-world NoiseFigure.qraw example
        parser = QspiceParser()
        # act
        definition = parser.parse_function_definition(".func NFDB(){20*LOG10(V(INOISE)@1/V(INOISE)@2)}")
        # assert — body is a binary expression whose LOG10 argument is a division of two selectors
        self.assertIsInstance(definition.body, BinaryOperationNode)
        self.assertEqual(definition.body.operator, BinaryOperator.MUL)
        log_call = definition.body.right
        self.assertIsInstance(log_call, FunctionCallNode)
        ratio = log_call.args[0]
        self.assertIsInstance(ratio, BinaryOperationNode)
        self.assertEqual(ratio.operator, BinaryOperator.DIV)
        self.assertIsInstance(ratio.left, StepSelectorNode)
        self.assertEqual(ratio.left.step_index, 1)
        self.assertIsInstance(ratio.right, StepSelectorNode)
        self.assertEqual(ratio.right.step_index, 2)
