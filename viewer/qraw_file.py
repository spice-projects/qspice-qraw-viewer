import logging
import mmap
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .variable import Variable, VariableType

logger = logging.getLogger(__name__)

# directory that contains the QML chart files
_QML_DIR = Path(__file__).parent / "qml"


class AbscissaScale(Enum):
    LINEAR = "lin"
    DECADE = "dec"
    OCTAVE = "oct"


@dataclass(frozen=True)
class ChartTypeSpec:
    """Static definition of a chart type: which QML file to load and which variable types it accepts."""

    # display name shown in the "Add Chart" context menu
    label: str
    # x axis unit
    xAxisUnit: str

    # QML file that defines the GraphsView layout for this chart type
    qml_file: Path
    # variable types accepted as series in this chart
    accepted_types: frozenset[VariableType]
    # when True, X data must be pre-transformed to log10 before passing to the series
    log_x: bool = False
    # maps VariableType to the QML function name used to add a series of that type
    # all chart types must provide an entry for each accepted type
    # excluded from hash/eq so ChartTypeSpec can be used as a dict key
    add_series_fn: dict[VariableType, str] = field(default_factory=dict, hash=False, compare=False)


# all available chart types — add new entries here as new chart types are created
TRANSIENT = ChartTypeSpec(
    label="Transient",
    xAxisUnit="s",

    qml_file=_QML_DIR / "transient_chart.qml",
    accepted_types=frozenset({VariableType.VOLTAGE, VariableType.CURRENT}),
    log_x=False,
    add_series_fn={
        VariableType.VOLTAGE: "addVoltageSeries",
        VariableType.CURRENT: "addCurrentSeries",
    },
)

AC = ChartTypeSpec(
    label="AC",
    xAxisUnit="Hz",

    qml_file=_QML_DIR / "ac_chart.qml",
    # AC analysis files can contain current probes (e.g. I(COUT) in VRM_Nyquist)
    accepted_types=frozenset({VariableType.VOLTAGE, VariableType.CURRENT}),
    log_x=True,
    add_series_fn={
        VariableType.VOLTAGE: "addVoltageSeries",
        VariableType.CURRENT: "addCurrentSeries",
    },
)

DC = ChartTypeSpec(
    label="DC",
    xAxisUnit="V",

    qml_file=_QML_DIR / "dc_chart.qml",
    # DC sweep and operating point files contain voltages, currents, power and parameters
    accepted_types=frozenset({VariableType.VOLTAGE, VariableType.CURRENT, VariableType.POWER, VariableType.PARAMETER}),
    log_x=False,
    add_series_fn={
        VariableType.VOLTAGE: "addVoltageSeries",
        VariableType.CURRENT: "addCurrentSeries",
        VariableType.POWER: "addCurrentSeries",
        VariableType.PARAMETER: "addVoltageSeries",
    },
)

# maps the mode-prefix keyword found inside «...» suggestion groups to a chart type
_MODE_TO_CHART: dict[str, ChartTypeSpec] = {
    "ac": AC,
    "tran": TRANSIENT,
    "dc": DC,
    "op": DC,
    "noise": AC,
}


def _chart_type_for_mode(mode: str) -> ChartTypeSpec | None:
    """Map a mode keyword from a plot suggestion (e.g. 'AC', 'tran') to a ChartTypeSpec.

    Returns None when the keyword is not recognised, so the caller can fall back
    to :func:`chart_type_for_file`.
    """
    return _MODE_TO_CHART.get(mode.lower())


def chart_type_for_file(qraw_file) -> ChartTypeSpec:
    """Return the correct chart type for a loaded file based on the X axis variable type."""
    # variables list must not be empty
    if not qraw_file.variables:
        return TRANSIENT
    # the abscissa (index 0) type unambiguously determines the chart layout
    x_type = qraw_file.variables[0].type
    if x_type == VariableType.FREQUENCY:
        return AC
    if x_type == VariableType.TIME:
        return TRANSIENT
    # VOLTAGE (DC transfer sweep) and PARAMETER (operating point sweep) both use the DC layout
    return DC


class PlotSuggestion:

    def __init__(self, chart_type: ChartTypeSpec, variables: list[Variable]):
        # fields
        self._chart_type = chart_type
        self._variables = variables

    @property
    def chart_type(self) -> ChartTypeSpec:
        return self._chart_type

    @property
    def variables(self) -> list[Variable]:
        return self._variables


class QRawFile:

    def __init__(self, filename: Path, title: str, date: str, plotname: str, complex: bool, stepped: bool, abscissa: str, abscissa_min: float, abscissa_max: float, abscissa_scale: AbscissaScale, command: str, plot_suggestion: str, num_points: int, variables: list[Variable], _mmap: mmap.mmap | None = None):
        # fields
        self._filename = filename
        self._title = title
        self._date = date
        self._plotname = plotname
        self._complex = complex
        self._stepped = stepped
        self._abscissa = abscissa
        self._abscissa_min = abscissa_min
        self._abscissa_max = abscissa_max
        self._abscissa_scale = abscissa_scale
        self._command = command
        self._plot_suggestion = plot_suggestion
        self._num_points = num_points
        self._variables = variables
        # keep the mmap alive for as long as this object exists — Variable._values arrays are zero-copy views into the mmap buffer; closing the mmap would invalidate all of them
        self._mmap = _mmap
        # calculated
        self._abscissa_points = len(variables[0].values) if variables else 0

    @property
    def filename(self) -> Path:
        return self._filename

    @property
    def title(self) -> str:
        return self._title

    @property
    def date(self) -> str:
        return self._date

    @property
    def plotname(self) -> str:
        return self._plotname

    @property
    def complex(self) -> bool:
        return self._complex

    @property
    def stepped(self) -> bool:
        return self._stepped

    @property
    def abscissa(self) -> str:
        return self._abscissa

    @property
    def abscissa_min(self) -> float:
        return self._abscissa_min

    @property
    def abscissa_max(self) -> float:
        return self._abscissa_max

    @property
    def abscissa_scale(self) -> AbscissaScale:
        return self._abscissa_scale

    @property
    def command(self) -> str:
        return self._command

    @property
    def plot_suggestion(self) -> str:
        return self._plot_suggestion

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def variables(self) -> list[Variable]:
        return self._variables

    @property
    def abscissa_points(self) -> int:
        return self._abscissa_points

    @property
    def plot_suggestions(self) -> list[PlotSuggestion]:
        # empty or whitespace-only suggestions string means no suggestions
        if not self.plot_suggestion.strip():
            return []
        # characters that indicate an expression rather than a plain variable name
        operators = set("/*+-^")
        # case-insensitive lookup from lowercased variable name to variable object
        by_name = {v.name.lower(): v for v in self.variables if v.index != 0}
        # plot suggestions
        plot_suggestions: list[PlotSuggestion] = []
        # extract each «...» group in order
        for group_text in re.findall("\xab(.*?)\xbb", self.plot_suggestion):
            # split group text into tokens; the first token may be a mode keyword
            tokens: list[str] = group_text.split()
            # chart type, defaults to chart type for file
            chart_type = chart_type_for_file(self)
            # resolve chart type from an optional leading mode keyword (ac, tran, dc …)
            if tokens and (resolved := _chart_type_for_mode(tokens[0])) is not None:
                # use chart type
                chart_type = resolved
                # remove mode token so the remaining tokens are variable names or expressions
                tokens = tokens[1:]
            # variables in group
            variables: list[Variable] = []
            # avoid duplicate variables within the same chart
            seen: set[int] = set()
            # loop tokens
            for token in tokens:
                # skip expression tokens (contain operator characters)
                if any(c in token for c in operators):
                    continue
                # case-insensitive match token to a variable name in the file (excluding the abscissa at index 0)
                variable = by_name.get(token.lower())
                # skip unmatched or already-added variables for this chart type
                if variable is None or variable.index in seen:
                    continue
                # add this variable to the chart type's list and mark it as seen
                seen.add(variable.index)
                # append variable
                variables.append(variable)
            # append to list
            plot_suggestions.append(PlotSuggestion(chart_type=chart_type, variables=variables))
        # exit
        return plot_suggestions

    @staticmethod
    def load(filename: str | Path):
        # load file
        path = Path(filename)
        if not path.exists():
            # log error
            logger.error("QRAW file not found: %s", path)
            # exit
            return None
        # measure time taken to load file
        start_time = time.perf_counter()
        try:
            # log information
            logger.info("Loading QRAW file: %s", path)
            # memory-map the file — the OS pages in only the regions that are actually read, so for a 300-variable file where only a few are displayed, the remaining columns are never loaded into physical RAM
            with open(path, "rb") as _file:
                data = mmap.mmap(_file.fileno(), 0, access=mmap.ACCESS_READ)
            # on POSIX (macOS/Linux) closing the fd after mmap() is safe — the OS keeps the mapping alive independently; the mmap object itself is stored in QRawFile to prevent GC
            # single-pass progressive parser: scan line by line until Binary: is reached
            header: dict[str, str] = {}
            variable_definitions: list[tuple[int, str, VariableType]] = []
            binary_offset = -1
            in_variables = False
            pos = 0
            # process file bytes
            while pos < len(data):
                # find \n
                newline = data.find(b"\n", pos)
                if newline == -1:
                    break
                # line: decode header text using Windows-1252
                line = data[pos:newline].decode("cp1252").strip()
                # advance position to next line
                pos = newline + 1
                # state machine to parse header and variables until binary section is reached
                if in_variables:
                    # check for end of variables section
                    if line == "Binary:":
                        # binary section starts at the current position in the file
                        binary_offset = pos
                        # exit loop
                        break
                    # parse variable line: expected format is "index\tname\ttype"
                    parts = line.split("\t")
                    # only parse lines with exactly 3 parts; skip malformed lines
                    if len(parts) == 3:
                        # find variable type from string by matching name
                        variable_type = next((vt for vt in VariableType if vt.value.name == parts[2]), None)
                        if variable_type is None:
                            # log information
                            logger.warning("Skipping variable with unknown type: %s", parts[2])
                            # skip malformed or unknown variable type
                            continue
                        # append variable to list
                        variable_definitions.append((int(parts[0]), parts[1], variable_type))
                        # next
                        continue
                # check this is the start of the variables section
                if line == "Variables:":
                    # state machine: next lines will contain variable definitions until "Binary:" is reached
                    in_variables = True
                    # next
                    continue
                # parse header line: expected format is "key: value"
                if ":" in line:
                    # split at the first colon to separate key and value; strip whitespace
                    key, _, value = line.partition(":")
                    # store in header dictionary
                    header[key.strip()] = value.strip()
            # validate that we found the binary section
            if binary_offset < 0:
                # log error
                logger.error("invalid QRAW file: Binary section not found")
                # exit
                return None
            # parse header values needed to interpret the binary data
            complex = "complex" in header.get("Flags", "").lower()
            stepped = "stepped" in header.get("Flags", "").lower()
            num_variables = int(header.get("No. Variables", 0))
            num_points = int(header.get("No. Points", "0").strip())
            # parse abscissa range and optional scale keyword (dec / oct / lin)
            abscissa_parts = header.get("Abscissa", "").split()
            abscissa_min = float(abscissa_parts[0]) if len(abscissa_parts) >= 1 else 0.0
            abscissa_max = float(abscissa_parts[1]) if len(abscissa_parts) >= 2 else 0.0
            abscissa_scale = AbscissaScale(abscissa_parts[2]) if len(abscissa_parts) >= 3 else AbscissaScale.LINEAR
            # check file contains vectors with complex numbers
            if complex:
                # complex files store the abscissa (index 0) as float64 and the remaining variables as complex128 in a structured array
                row_dtype = np.dtype([("abscissa", "<f8"), ("data", "<c16", num_variables - 1)])
                # parse binary data into a structured array with separate fields for abscissa and data variables; the abscissa is stored in a separate field to allow it to be read as float64 while the data variables are stored as complex128
                matrix = np.frombuffer(data, dtype=row_dtype, offset=binary_offset)
                # abscissa definition
                abscissa_definition = variable_definitions[0]
                # abscissa variable
                abscissa_variable = Variable(index=abscissa_definition[0], name=abscissa_definition[1], type=abscissa_definition[2], values=matrix["abscissa"])
                # all variables
                variables = [abscissa_variable] + [Variable(index=idx, name=name, type=type, values=matrix["data"][:, idx - 1]) for idx, name, type in variable_definitions[1:]]
            else:
                # real files store all variables as float64 in a uniform layout; parse binary data into a 2D array with shape (num_points, num_variables)
                flat = np.frombuffer(data, dtype="<f8", offset=binary_offset)
                # reshape flat array into a 2D array with shape (num_points, num_variables); the data is stored in row-major order, so each row corresponds to a point and each column corresponds to a variable
                matrix = flat.reshape(num_points, num_variables)
                # extract variables
                variables = [Variable(index=idx, name=name, type=type, values=matrix[:, idx]) for idx, name, type in variable_definitions]
            # check this is a stepped file
            if stepped:
                # abscissa variable data
                abscissa_variable_data = variables[0].values
                # calculate the period, O(n) at C speed
                period = np.argmax(np.isclose(abscissa_variable_data[1:], abscissa_variable_data[0], rtol=1e-6)) + 1
                # calculate the number of steps and points per step based on the period
                steps = num_points // period
                # recalculate abscissa (no need to store additional data)
                abscissa_variable = Variable(index=0, name=variables[0].name, type=variables[0].type, values=abscissa_variable_data[:period], steps=steps)
                # update all variables with step information
                variables = [abscissa_variable] + [Variable(index=v.index, name=v.name, type=v.type, values=v.values, steps=steps) for v in variables[1:]]
            # process scale (x axis)
            if abscissa_scale == AbscissaScale.DECADE:
                # apply log10 only once per file
                variables[0] = Variable(index=0, name=variables[0].name, type=variables[0].type, values=np.log10(variables[0].values), steps=variables[0].steps)
            elif abscissa_scale == AbscissaScale.OCTAVE:
                # apply log2 only once per file
                variables[0] = Variable(index=0, name=variables[0].name, type=variables[0].type, values=np.log2(variables[0].values), steps=variables[0].steps)
            # create QRawFile instance with parsed header, variables, and binary data; pass the mmap so it stays alive for the lifetime of the QRawFile — Variable arrays are views into it
            return QRawFile(filename=path, title=header.get("Title", ""), date=header.get("Date", ""), plotname=header.get("Plotname", ""), complex=complex, stepped=stepped, abscissa=header.get("Abscissa", ""), abscissa_min=abscissa_min, abscissa_max=abscissa_max, abscissa_scale=abscissa_scale, command=header.get("Command", ""), plot_suggestion=header.get("Plot Suggestion(s)", ""), num_points=num_points, variables=variables, _mmap=data)
        finally:
            # log information
            logger.info("Finished loading QRAW file: %s, latency: %f seconds", path, time.perf_counter() - start_time)
