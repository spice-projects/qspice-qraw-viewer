import logging

from .expression import Expression
from .expression_parser import ExpressionParser
from .expression_evaluator import ExpressionEvaluator

logger = logging.getLogger(__name__)


class ExpressionManager:

    def __init__(self, expressions: list[Expression]):
        # create expression context
        self._context: dict[str, Expression] = {expression.name: expression for expression in expressions}
        # initialize the parser and evaluator instances
        self._parser = ExpressionParser()
        self._evaluator = ExpressionEvaluator()

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
                # parse the expression string into an AST
                ast = self._parser.parse(expression)
                # evaluate the AST using the provided context
                result = self._evaluator.evaluate(ast, self._context, name, "expression manager")
                # update context with the evaluated expression for future reference
                self._context[key] = result
            except ValueError as e:
                # log information
                logger.warning("Failed to evaluate expression %r: %s", expression, e)
        # exit
        return result
