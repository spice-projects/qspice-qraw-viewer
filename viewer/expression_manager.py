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
        return list(self._context.values())

    def evaluate(self, expr_str: str) -> Expression | None:
        # check expression has been evaluated before
        result = self._context.get(expr_str, None)
        if result is None:
            try:
                # parse the expression string into an AST
                ast = self._parser.parse(expr_str)
                # evaluate the AST using the provided context
                result = self._evaluator.evaluate(ast, self._context, expr_str)
                # update context with the evaluated expression for future reference
                self._context[expr_str] = result
            except Exception as e:
                # log information
                logger.warning("Failed to evaluate expression %r: %s", expr_str, e)
        # exit
        return result
