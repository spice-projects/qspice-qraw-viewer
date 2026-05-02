from unittest import TestCase

from viewer.qspice_language.nodes import BinaryOperationNode, BinaryOperator, FunctionCallNode, IdentifierNode, NumberNode, UnaryOperationNode, UnaryOperator
from viewer.qspice_language.parser import QspiceParser


class TestQspiceParserParity(TestCase):
    """Port all test cases from expression_parser_test.py."""

    # ------------------------------------------------------------------ #
    # Literals                                                             #
    # ------------------------------------------------------------------ #

    def test_parse_integer_literal(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("10")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertEqual(node.text, "10")

    def test_parse_float_literal(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("3.14")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertEqual(node.text, "3.14")

    def test_parse_scientific_notation(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("1e-3")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertEqual(node.text, "1e-3")

    # ------------------------------------------------------------------ #
    # Identifiers and SPICE probes                                        #
    # ------------------------------------------------------------------ #

    def test_parse_bare_identifier(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("Vout")
        # assert
        self.assertIsInstance(node, IdentifierNode)
        self.assertEqual(node.name, "Vout")

    def test_parse_voltage_probe_single_node(self):
        # arrange — V(out) should parse as a voltage probe reference
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out)")
        # assert — new parser treats this as a function call to probe handler
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "V")
        self.assertEqual(len(node.args), 1)
        self.assertIsInstance(node.args[0], IdentifierNode)
        self.assertEqual(node.args[0].name, "out")

    def test_parse_voltage_probe_two_nodes(self):
        # arrange — V(a, b) is a differential probe reference
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(R1, 0)")
        # assert — parses as function call with two arguments
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "V")
        self.assertEqual(len(node.args), 2)
        self.assertIsInstance(node.args[0], IdentifierNode)
        self.assertEqual(node.args[0].name, "R1")
        self.assertIsInstance(node.args[1], NumberNode)
        self.assertEqual(node.args[1].text, "0")

    def test_parse_current_probe(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("I(R1)")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "I")

    def test_parse_device_terminal_current(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("Id(J1)")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "Id")

    def test_parse_bus_node_variable(self):
        # arrange — QSPICE allows array/bus notation like V(signal[3])
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(outd[3])")
        # assert — parses as probe with array index in argument
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "V")

    # ------------------------------------------------------------------ #
    # Function calls                                                       #
    # ------------------------------------------------------------------ #

    def test_parse_db_function(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("db(V(out))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "db")
        self.assertEqual(len(node.args), 1)
        self.assertIsInstance(node.args[0], FunctionCallNode)
        self.assertEqual(node.args[0].name, "V")

    def test_parse_abs_function(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("abs(I(R1))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "abs")

    def test_parse_sqrt_function(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("sqrt(V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "sqrt")

    # ------------------------------------------------------------------ #
    # Binary operators                                                     #
    # ------------------------------------------------------------------ #

    def test_parse_addition(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out) + V(in)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.ADD)
        self.assertIsInstance(node.left, FunctionCallNode)
        self.assertIsInstance(node.right, FunctionCallNode)

    def test_parse_subtraction(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out) - V(in)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.SUB)

    def test_parse_multiplication(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("10 * V(R1)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left, NumberNode)
        self.assertEqual(node.left.text, "10")
        self.assertIsInstance(node.right, FunctionCallNode)

    def test_parse_division(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out) / I(R1)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.DIV)

    def test_parse_power_caret(self):
        # arrange — caret (^) is QSPICE power operator
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.POW)

    def test_parse_power_double_star(self):
        # arrange — ** is also a valid power operator
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(out) ** 2")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.POW)

    # ------------------------------------------------------------------ #
    # Operator precedence                                                 #
    # ------------------------------------------------------------------ #

    def test_precedence_mul_before_add(self):
        # arrange — "a + b * c" should parse as "a + (b * c)"
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(a) + 2 * V(b)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.ADD)
        self.assertIsInstance(node.right, BinaryOperationNode)
        self.assertEqual(node.right.operator, BinaryOperator.MUL)

    def test_precedence_power_before_mul(self):
        # arrange — "a * b ^ 2" should parse as "a * (b ^ 2)"
        parser = QspiceParser()
        # act
        node = parser.parse_expression("2 * V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.right, BinaryOperationNode)
        self.assertEqual(node.right.operator, BinaryOperator.POW)

    # ------------------------------------------------------------------ #
    # Unary operators                                                      #
    # ------------------------------------------------------------------ #

    def test_parse_unary_negation(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("-V(out)")
        # assert
        self.assertIsInstance(node, UnaryOperationNode)
        self.assertEqual(node.operator, UnaryOperator.NEG)
        self.assertIsInstance(node.operand, FunctionCallNode)

    def test_parse_unary_plus(self):
        # arrange — +x is a valid unary operator
        parser = QspiceParser()
        # act
        node = parser.parse_expression("+V(out)")
        # assert
        self.assertIsInstance(node, UnaryOperationNode)
        self.assertEqual(node.operator, UnaryOperator.POS)

    # ------------------------------------------------------------------ #
    # Parentheses and grouping                                            #
    # ------------------------------------------------------------------ #

    def test_parse_parenthesised_addition(self):
        # arrange
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(V(out) + V(in)) * 2")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left, BinaryOperationNode)
        self.assertEqual(node.left.operator, BinaryOperator.ADD)

    def test_parse_db_ratio(self):
        # arrange — "db(V(out)/V(in))" from the issue examples
        parser = QspiceParser()
        # act
        node = parser.parse_expression("db(V(out)/V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "db")
        inner = node.args[0]
        self.assertIsInstance(inner, BinaryOperationNode)
        self.assertEqual(inner.operator, BinaryOperator.DIV)

    # ------------------------------------------------------------------ #
    # Error cases                                                          #
    # ------------------------------------------------------------------ #

    def test_parse_error_unexpected_character(self):
        # arrange
        parser = QspiceParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse_expression("V(out) % 2")

    def test_parse_error_trailing_token(self):
        # arrange
        parser = QspiceParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse_expression("V(out) 2")

    def test_parse_error_empty_expression(self):
        # arrange
        parser = QspiceParser()
        # act / assert
        with self.assertRaises(ValueError):
            parser.parse_expression("()")

    # ------------------------------------------------------------------ #
    # QSPICE alias expressions (real-world from .qraw files)               #
    # ------------------------------------------------------------------ #

    def test_parse_alias_bare_variable_reference(self):
        # arrange — from '.alias Freq Frequency' in VRM_GainBW.qraw
        parser = QspiceParser()
        # act
        node = parser.parse_expression("Frequency")
        # assert
        self.assertIsInstance(node, IdentifierNode)
        self.assertEqual(node.name, "Frequency")

    def test_parse_alias_omega_expression(self):
        # arrange — from '.alias Omega (2*pi*Frequency)' in VRM_GainBW.qraw
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(2*pi*Frequency)")
        # assert
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left, BinaryOperationNode)
        self.assertEqual(node.left.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertEqual(node.left.left.text, "2")
        self.assertIsInstance(node.left.right, IdentifierNode)
        self.assertEqual(node.left.right.name, "pi")
        self.assertIsInstance(node.right, IdentifierNode)
        self.assertEqual(node.right.name, "Frequency")

    def test_parse_alias_conductance_times_voltage(self):
        # arrange — from '.alias I(R4) (1mho*V(out,0))' in Buck_COT_TRAN.qraw
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(1mho*V(out,0))")
        # assert — outer is multiplication
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        # left side: 1*mho
        self.assertIsInstance(node.left, BinaryOperationNode)
        self.assertEqual(node.left.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertEqual(node.left.left.text, "1")
        self.assertIsInstance(node.left.right, IdentifierNode)
        self.assertEqual(node.left.right.name, "mho")
        # right side: V(out, 0)
        self.assertIsInstance(node.right, FunctionCallNode)
        self.assertEqual(node.right.name, "V")

    def test_parse_alias_scientific_conductance_times_voltage(self):
        # arrange — from '.alias I(RCOT) (1e-05mho*V(in,n06))' in Buck_COT_TRAN.qraw
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(1e-05mho*V(in,n06))")
        # assert — outer is multiplication
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        # left side: 1e-05*mho
        self.assertIsInstance(node.left, BinaryOperationNode)
        self.assertEqual(node.left.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.left.left, NumberNode)
        self.assertIn("1e-05", node.left.left.text)
        self.assertIsInstance(node.left.right, IdentifierNode)
        self.assertEqual(node.left.right.name, "mho")
        # right side: V(in, n06)
        self.assertIsInstance(node.right, FunctionCallNode)

    def test_parse_alias_bullet_hierarchy_separator_in_probe(self):
        # arrange — QSPICE uses U+2022 (bullet •) as hierarchy separator
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(a\u2022b\u2022c)")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "V")

    def test_parse_alias_hash_in_node_name(self):
        # arrange — QSPICE uses '#' for special quantities like '#current'
        parser = QspiceParser()
        # act
        node = parser.parse_expression("V(d3\u2022x1\u2022xu302#current)")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "V")

    def test_parse_alias_bullet_with_digit_start_node(self):
        # arrange — node names can start with a digit when decoded from cp1252
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(2.5mho*V(26a\u2022x1\u2022xt301,0))")
        # assert — outer is multiplication
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.right, FunctionCallNode)

    def test_parse_alias_bullet_in_current_probe_arg(self):
        # arrange — from '.alias I(F_F1•X_F1•XU305) (1*I(VF_F1•X_F1•XU305))' in test.qraw
        parser = QspiceParser()
        # act
        node = parser.parse_expression("(1*I(VF_F1\u2022X_F1\u2022XU305))")
        # assert — outer is multiplication
        self.assertIsInstance(node, BinaryOperationNode)
        self.assertEqual(node.operator, BinaryOperator.MUL)
        self.assertIsInstance(node.right, FunctionCallNode)
        self.assertEqual(node.right.name, "I")
