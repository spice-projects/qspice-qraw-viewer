from unittest import TestCase

from viewer.qspice_language.lexer import QspiceLexer
from viewer.qspice_language.tokens import TokenKind


class TestQspiceLexer(TestCase):

    def test_tokenize_func_declaration(self):
        # arrange
        lexer = QspiceLexer()
        # act
        tokens = lexer.tokenize(".func gain(x, y) {x+y}")
        # assert
        self.assertEqual([token.kind for token in tokens], [
            TokenKind.DIRECTIVE,
            TokenKind.IDENTIFIER,
            TokenKind.LPAREN,
            TokenKind.IDENTIFIER,
            TokenKind.COMMA,
            TokenKind.IDENTIFIER,
            TokenKind.RPAREN,
            TokenKind.LBRACE,
            TokenKind.IDENTIFIER,
            TokenKind.PLUS,
            TokenKind.IDENTIFIER,
            TokenKind.RBRACE,
            TokenKind.EOF,
        ])
        self.assertEqual([token.text for token in tokens[:-1]], [".func", "gain", "(", "x", ",", "y", ")", "{", "x", "+", "y", "}"])

    def test_tokenize_number_with_exponent_and_suffix(self):
        # arrange — suffixes are now tokenized separately for implicit multiplication support
        lexer = QspiceLexer()
        # act
        tokens = lexer.tokenize("1e-05meg")
        # assert — NUMBER followed by IDENTIFIER
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].kind, TokenKind.NUMBER)
        self.assertEqual(tokens[0].text, "1e-05")
        self.assertEqual(tokens[1].kind, TokenKind.IDENTIFIER)
        self.assertEqual(tokens[1].text, "meg")

    def test_tokenize_multi_character_operators(self):
        # arrange
        lexer = QspiceLexer()
        # act
        tokens = lexer.tokenize("a<=b && c!=d || e>=f ? g : h ** 2")
        # assert
        self.assertEqual([token.kind for token in tokens], [
            TokenKind.IDENTIFIER,
            TokenKind.LESS_EQUAL,
            TokenKind.IDENTIFIER,
            TokenKind.LOGICAL_AND,
            TokenKind.IDENTIFIER,
            TokenKind.BANG_EQUAL,
            TokenKind.IDENTIFIER,
            TokenKind.LOGICAL_OR,
            TokenKind.IDENTIFIER,
            TokenKind.GREATER_EQUAL,
            TokenKind.IDENTIFIER,
            TokenKind.QUESTION,
            TokenKind.IDENTIFIER,
            TokenKind.COLON,
            TokenKind.IDENTIFIER,
            TokenKind.POWER,
            TokenKind.NUMBER,
            TokenKind.EOF,
        ])

    def test_tokenize_records_source_spans(self):
        # arrange
        lexer = QspiceLexer()
        # act
        tokens = lexer.tokenize("foo + 12")
        # assert
        self.assertEqual((tokens[0].start, tokens[0].end), (0, 3))
        self.assertEqual((tokens[1].start, tokens[1].end), (4, 5))
        self.assertEqual((tokens[2].start, tokens[2].end), (6, 8))

    def test_tokenize_conductance_alias_style_number(self):
        # arrange — implicit multiplication: "1mho" tokenizes as NUMBER + IDENTIFIER
        lexer = QspiceLexer()
        # act
        tokens = lexer.tokenize("1mho*V(out,0)")
        # assert
        self.assertEqual([token.kind for token in tokens], [
            TokenKind.NUMBER,
            TokenKind.IDENTIFIER,
            TokenKind.STAR,
            TokenKind.IDENTIFIER,
            TokenKind.LPAREN,
            TokenKind.IDENTIFIER,
            TokenKind.COMMA,
            TokenKind.NUMBER,
            TokenKind.RPAREN,
            TokenKind.EOF,
        ])
        self.assertEqual(tokens[0].text, "1")
        self.assertEqual(tokens[1].text, "mho")

    def test_tokenize_invalid_character_raises_error(self):
        # arrange
        lexer = QspiceLexer()
        # act / assert
        with self.assertRaises(ValueError):
            lexer.tokenize("x % y")
