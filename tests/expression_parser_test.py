from unittest import TestCase

from viewer.expression_node import BinaryOp, BinaryOpNode, FunctionCallNode, NumberNode, UnaryOp, UnaryOpNode, VariableRefNode
from viewer.expression_parser import ExpressionParser


class TestExpressionParser(TestCase):

    def test_parse_integer_literal(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("10")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertEqual(node.value, 10.0)

    def test_parse_float_literal(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("3.14")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertAlmostEqual(node.value, 3.14)

    def test_parse_scientific_notation(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("1e-3")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertAlmostEqual(node.value, 0.001)

    def test_parse_bare_identifier(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("Vout")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "Vout")

    def test_parse_voltage_probe(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(out)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(out)")

    def test_parse_voltage_probe_two_nodes(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(R1, 0)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(R1, 0)")

    def test_parse_current_probe(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("I(R1)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "I(R1)")

    def test_parse_device_terminal_current(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("Id(J1)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "Id(J1)")

    def test_parse_bus_node_variable(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(outd[3])")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(outd[3])")

    def test_parse_db_function(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("db(V(out))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name.lower(), "db")
        self.assertEqual(len(node.args), 1)
        self.assertIsInstance(node.args[0], VariableRefNode)
        self.assertEqual(node.args[0].name, "V(out)")

    def test_parse_abs_function(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("abs(I(R1))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "abs")

    def test_parse_sqrt_function(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("sqrt(V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "sqrt")

    def test_parse_addition(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(out) + V(in)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.ADD)
        self.assertIsInstance(node.left, VariableRefNode)
        self.assertIsInstance(node.right, VariableRefNode)

    def test_parse_subtraction(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(out) - V(in)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.SUB)

    def test_parse_multiplication(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("10 * V(R1)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, NumberNode)
        self.assertEqual(node.left.value, 10.0)
        self.assertIsInstance(node.right, VariableRefNode)

    def test_parse_division(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(out) / I(R1)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.DIV)

    def test_parse_power(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.POW)

    def test_precedence_mul_before_add(self):
        # arrange — "a + b * c" should parse as "a + (b * c)"
        parser = ExpressionParser()
        # act
        node = parser.parse("V(a) + 2 * V(b)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.ADD)
        self.assertIsInstance(node.right, BinaryOpNode)
        self.assertEqual(node.right.op, BinaryOp.MUL)

    def test_precedence_power_before_mul(self):
        # arrange — "a * b ^ 2" should parse as "a * (b ^ 2)"
        parser = ExpressionParser()
        # act
        node = parser.parse("2 * V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.right, BinaryOpNode)
        self.assertEqual(node.right.op, BinaryOp.POW)

    def test_parse_unary_negation(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("-V(out)")
        # assert
        self.assertIsInstance(node, UnaryOpNode)
        self.assertEqual(node.op, UnaryOp.NEG)
        self.assertIsInstance(node.operand, VariableRefNode)

    def test_parse_parenthesised_addition(self):
        # arrange
        parser = ExpressionParser()
        # act
        node = parser.parse("(V(out) + V(in)) * 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, BinaryOp.ADD)

    def test_parse_db_ratio(self):
        # arrange — "db(V(out)/V(in))" from the issue examples
        parser = ExpressionParser()
        # act
        node = parser.parse("db(V(out)/V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name.lower(), "db")
        inner = node.args[0]
        self.assertIsInstance(inner, BinaryOpNode)
        self.assertEqual(inner.op, BinaryOp.DIV)

    def test_parse_error_unexpected_character(self):
        # arrange
        parser = ExpressionParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse("V(out) @ 2")

    def test_parse_error_trailing_token(self):
        # arrange
        parser = ExpressionParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse("V(out) 2")

    def test_parse_error_empty_expression(self):
        # arrange
        parser = ExpressionParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse("()")
