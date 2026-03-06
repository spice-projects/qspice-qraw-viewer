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
        self._complex = data.dtype == np.complex128
        self._source = source
        # contiguos data in memory
        self._contiguous_data: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Display name of the expression (e.g. ``"V(R1)"`` or ``"10 * V(R1)"``."""
        return self._name

    @property
    def data(self) -> np.ndarray:
        """Evaluated data array, one value per simulation point."""
        return self._data

    @property
    def values(self) -> np.ndarray:
        """Evaluated data array, one value per simulation point, guaranteed to be contiguous in memory for efficient plotting."""
        # check we have calculated contiguous data already
        if self._contiguous_data is None:
            # calculate contiguous data from the original data array
            self._contiguous_data = self._data if self._data.flags.c_contiguous else np.ascontiguousarray(self._data)
        # exit
        return self._contiguous_data

    @property
    def unit(self) -> str:
        """Physical unit propagated through the expression tree (e.g. ``"V"``, ``"A"``, ``"W"``)."""
        return self._unit

    @property
    def complex(self) -> bool:
        """Indicates whether the expression evaluates to a complex number."""
        return self._complex

    @property
    def source(self) -> str | None:
        """Original source expression string, if available."""
        return self._source
