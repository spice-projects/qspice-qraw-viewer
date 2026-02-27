from enum import Enum

import numpy as np


class VariableTypeInformation:

    def __init__(self, name: str, unit: str):
        self._name = name
        self._unit = unit

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> str:
        return self._unit


class VariableType(Enum):
    FREQUENCY = VariableTypeInformation("frequency", "Hz")
    VOLTAGE = VariableTypeInformation("voltage", "V")
    CURRENT = VariableTypeInformation("current", "A")
    TIME = VariableTypeInformation("time", "s")
    POWER = VariableTypeInformation("power", "W")
    PARAMETER = VariableTypeInformation("parameter", "")
    PHASE = VariableTypeInformation("phase", "°")


class Variable:

    def __init__(self, index: int, name: str, type: VariableType, values: np.ndarray, steps: int = 1, magnitude: Variable | None = None, phase: Variable | None = None):
        # fields
        self._index = index
        self._name = name
        self._type = type
        self._values = values
        self._steps = steps
        self._step_length = len(values) // steps
        # computed values
        self._contiguous_values: np.ndarray | None = None
        self._magnitude: Variable | None = magnitude
        self._phase: Variable | None = phase

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> VariableType:
        return self._type

    @property
    def values(self) -> np.ndarray:
        # calculate contiguous values if we haven't already
        if self._contiguous_values is None:
            self._contiguous_values = np.ascontiguousarray(self._values)
        # exit
        return self._contiguous_values

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def complex(self) -> bool:
        return self._values.dtype == np.complex128

    def step_values(self, step: int) -> np.ndarray:
        # validate step index
        if step < 0 or step >= self._steps:
            raise IndexError(f"step index {step} out of range for variable with {self._steps} steps")
        # return a view into the appropriate slice of the values array for this step; this is a zero-copy operation so it's efficient even for large arrays
        return self.values[step * self._step_length: (step + 1) * self._step_length]

    @property
    def magnitude(self) -> Variable:
        # variable must be complex
        if self._values.dtype != np.complex128:
            raise ValueError("cannot calculate magnitude of a non-complex variable")
        # check we have calculated the magnitude for this variable already; if so, return it
        if self._magnitude is None:
            # calculate magnitude from the complex data vector; this is an expensive operation, so we only want to do it once per variable
            self._magnitude = Variable(self._index, f"abs({self._name})", self._type, np.abs(self.values), self._steps)
        # exit
        return self._magnitude

    @property
    def phase(self) -> Variable:
        # variable must be complex
        if self._values.dtype != np.complex128:
            raise ValueError("cannot calculate phase of a non-complex variable")
        # check we have calculated the phase for this variable already; if so, return it
        if self._phase is None:
            # calculate phase from the complex data vector; this is an expensive operation, so we only want to do it once per variable
            self._phase = Variable(self._index, f"angle({self._name})", VariableType.PHASE, np.angle(self.values, True), self._steps)
        # exit
        return self._phase
