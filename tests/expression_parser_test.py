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

    def test_parse_alias_bare_variable_reference(self):
        # arrange — from '.alias Freq Frequency' in VRM_GainBW.qraw / Buck_COT_Bode.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("Frequency")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "Frequency")

    def test_parse_alias_omega_expression(self):
        # arrange — from '.alias Omega (2*pi*Frequency)' in VRM_GainBW.qraw / Buck_COT_Bode.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("(2*pi*Frequency)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, BinaryOp.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertEqual(node.left.left.value, 2.0)
        self.assertIsInstance(node.left.right, VariableRefNode)
        self.assertEqual(node.left.right.name, "pi")
        self.assertIsInstance(node.right, VariableRefNode)
        self.assertEqual(node.right.name, "Frequency")

    def test_parse_alias_conductance_times_voltage(self):
        # arrange — from '.alias I(R4) (1mho*V(out,0))' in Buck_COT_TRAN.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("(1mho*V(out,0))")
        # assert — outer is multiplication, left is implicit 1*mho, right is V(out, 0)
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, BinaryOp.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertEqual(node.left.left.value, 1.0)
        self.assertIsInstance(node.left.right, VariableRefNode)
        self.assertEqual(node.left.right.name, "mho")
        self.assertIsInstance(node.right, VariableRefNode)
        # parser normalizes argument spacing: V(out,0) → "V(out, 0)" (comma + space)
        self.assertEqual(node.right.name, "V(out, 0)")

    def test_parse_alias_scientific_conductance_times_voltage(self):
        # arrange — from '.alias I(RCOT) (1e-05mho*V(in,n06))' in Buck_COT_TRAN.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("(1e-05mho*V(in,n06))")
        # assert — outer is multiplication, left is implicit 1e-05*mho, right is V(in, n06)
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, BinaryOp.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertAlmostEqual(node.left.left.value, 1e-05)
        self.assertIsInstance(node.left.right, VariableRefNode)
        self.assertEqual(node.left.right.name, "mho")
        self.assertIsInstance(node.right, VariableRefNode)
        # parser normalizes argument spacing: V(in,n06) → "V(in, n06)" (comma + space)
        self.assertEqual(node.right.name, "V(in, n06)")

    def test_parse_alias_bullet_hierarchy_separator_in_probe(self):
        # arrange — QSPICE uses U+2022 (bullet •) as a hierarchy separator in node names,
        # e.g. from '.alias I(R1•XU1) (1mho*V(a•b•c,0))' in test.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("V(a\u2022b\u2022c)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(a\u2022b\u2022c)")

    def test_parse_alias_hash_in_node_name(self):
        # arrange — QSPICE uses '#' to denote special quantities like '#current',
        # e.g. from '.alias I(D3•X1•XU302) (1mho*V(d3•x1•xu302#current))' in test.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("V(d3\u2022x1\u2022xu302#current)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(d3\u2022x1\u2022xu302#current)")

    def test_parse_alias_bullet_with_digit_start_node(self):
        # arrange — from '.alias I(RS8•X1•XT301) (2.5mho*V(26a•x1•xt301,0))' in test.qraw;
        # node names can start with a digit when decoded from cp1252
        parser = ExpressionParser()
        # act
        node = parser.parse("(2.5mho*V(26a\u2022x1\u2022xt301,0))")
        # assert — outer is multiplication, left is 2.5*mho, right is the probe
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.right, VariableRefNode)
        # digit-starting tokens (NUMBER) and IDENT tokens are joined verbatim in raw arg context
        self.assertEqual(node.right.name, "V(26a\u2022x1\u2022xt301, 0)")

    def test_parse_alias_bullet_in_current_probe_arg(self):
        # arrange — from '.alias I(F_F1•X_F1•XU305) (1*I(VF_F1•X_F1•XU305))' in test.qraw
        parser = ExpressionParser()
        # act
        node = parser.parse("(1*I(VF_F1\u2022X_F1\u2022XU305))")
        # assert — outer is multiplication, right is the current probe with bullet path
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.right, VariableRefNode)
        self.assertEqual(node.right.name, "I(VF_F1\u2022X_F1\u2022XU305)")
