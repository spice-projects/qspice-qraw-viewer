import numpy as np


class Expression:
    """A named, evaluated expression with a data array and a propagated unit.

    Instances are produced by :class:`~viewer.expression_evaluator.ExpressionEvaluator`
    and represent any computed trace — whether derived from a .alias directive,
    typed interactively by the user, or loaded directly from the QRAW file.
    """

    def __init__(self, name: str, data: np.ndarray, unit: str, source: str | None = None):
        # fields
        self._name = name
        self._data = data
        self._unit = unit
        self._source = source

    @property
    def name(self) -> str:
        """Display name of the expression (e.g. ``"V(R1)"`` or ``"10 * V(R1)"``."""
        return self._name

    @property
    def data(self) -> np.ndarray:
        """Evaluated data array, one value per simulation point."""
        return self._data

    @property
    def unit(self) -> str:
        """Physical unit propagated through the expression tree (e.g. ``"V"``, ``"A"``, ``"W"``)."""
        return self._unit

    @property
    def source(self) -> str | None:
        """Original source expression string, if available."""
        return self._source
