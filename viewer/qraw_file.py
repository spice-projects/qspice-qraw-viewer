import logging
import mmap
import re
import time
from enum import Enum
from pathlib import Path

import numpy as np

from .expression import Expression
from .expression_manager import ExpressionManager

logger = logging.getLogger(__name__)


class AbscissaScale(Enum):
    LINEAR = "lin"
    DECADE = "dec"
    OCTAVE = "oct"


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


class PlotSuggestion:

    def __init__(self, chart_type: str, expressions: list[Expression]):
        # fields
        self._chart_type = chart_type
        self._expressions = expressions

    @property
    def chart_type(self) -> str:
        return self._chart_type

    @property
    def expressions(self) -> list[Expression]:
        return self._expressions


def _process_step(abscissa: Expression, num_points: int) -> tuple[int, Expression]:
    # calculate the period, O(n) at C speed
    period = np.argmax(np.isclose(abscissa.data[1:], abscissa.data[0], rtol=1e-6)) + 1
    # calculate the number of steps and points per step based on the period
    steps = num_points // period
    # number of steps and points per step
    return steps, Expression(abscissa.name, abscissa.data[:period], abscissa.unit, abscissa.source)


def _process_scale(abscissa: Expression, scale: AbscissaScale) -> Expression:
    # log10
    if scale == AbscissaScale.DECADE:
        return Expression(abscissa.name, np.log10(abscissa.data), abscissa.unit, abscissa.source)
    # log2
    if scale == AbscissaScale.OCTAVE:
        return Expression(abscissa.name, np.log2(abscissa.data), abscissa.unit, abscissa.source)
    # linear scale doesn't modify the abscissa
    return abscissa


_MODE_TO_CHART: dict[str, str] = {
    "ac": "AC",
    "tran": "TRANSIENT",
    "dc": "DC",
    "op": "DC",
    "noise": "AC",
}


def _chart_type_for_file(abscissa: Expression) -> str:
    # type unambiguously determines the chart layout
    if abscissa.unit == "Hz":
        return "AC"
    if abscissa.unit == "s":
        return "TRANSIENT"
    # VOLTAGE (DC transfer sweep) and PARAMETER (operating point sweep) both use the DC layout
    return "DC"


class QRawFile:

    def __init__(self, filename: Path, title: str, date: str, plotname: str, complex: bool, steps: int, abscissa: Expression, abscissa_scale: AbscissaScale, command: str, plot_suggestion: str, expression_manager: ExpressionManager, _mmap: mmap.mmap | None = None):
        # fields
        self._filename = filename
        self._title = title
        self._date = date
        self._plotname = plotname
        self._complex = complex
        self._steps = steps
        self._abscissa = abscissa
        self._abscissa_scale = abscissa_scale
        self._command = command
        self._plot_suggestion = plot_suggestion
        self._expression_manager = expression_manager
        # keep the mmap alive for as long as this object exists — Variable._values arrays are zero-copy views into the mmap buffer; closing the mmap would invalidate all of them
        self._mmap = _mmap
        # calculated
        self._abscissa_points = len(abscissa.data)
        self._plot_suggestions: list[PlotSuggestion] | None = None

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
    def steps(self) -> int:
        return self._steps

    @property
    def abscissa(self) -> Expression:
        return self._abscissa

    @property
    def abscissa_scale(self) -> AbscissaScale:
        return self._abscissa_scale

    @property
    def command(self) -> str:
        return self._command

    @property
    def expression_manager(self) -> ExpressionManager:
        return self._expression_manager

    def get_plot_suggestions(self) -> list[PlotSuggestion]:
        # empty or whitespace-only suggestions string means no suggestions
        if not self._plot_suggestion.strip():
            return []
        # calculate plot suggestions
        if self._plot_suggestions is None:
            # file chart type
            file_chart_type = _chart_type_for_file(self._abscissa)
            # plot suggestions
            self._plot_suggestions = []
            # extract each «...» group in order, ``\xab``/``\xbb`` are the «» characters; using a raw string avoids accidental escaping.
            for group_text in re.findall(r"\xab(.*?)\xbb", self._plot_suggestion):
                # split group text into tokens; the first token may be a mode keyword
                tokens: list[str] = group_text.split()
                # chart type, defaults to chart type for file
                chart_type = file_chart_type
                # resolve chart type from an optional leading mode keyword (ac, tran, dc …)
                if tokens and (resolved := _MODE_TO_CHART.get(tokens[0].lower())) is not None:
                    # use chart type
                    chart_type = resolved
                    # remove mode token so the remaining tokens are variable names or expressions
                    tokens = tokens[1:]
                # expressions in group
                expressions: list[Expression] = []
                # avoid duplicate expressions within the same chart
                seen: set[Expression] = set()
                # loop tokens
                for token in tokens:
                    # evaluate expression
                    expression = self._expression_manager.evaluate(token)
                    # skip unmatched or already-added variables for this chart type
                    if expression is None or expression in seen:
                        continue
                    # add this expression to the chart type's list and mark it as seen
                    seen.add(expression)
                    # append expression
                    expressions.append(expression)
                # append to list
                self._plot_suggestions.append(PlotSuggestion(chart_type=chart_type, expressions=expressions))
        # exit
        return self._plot_suggestions

    @staticmethod
    def load(filename: str | Path) -> "QRawFile" | None:
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
            # aliases
            aliasses: dict[str, str] = {}
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
                # alias
                if line.startswith(".alias"):
                    # split line (space)
                    parts = line.split()
                    # should be three parts ".alias Name Expression"
                    if len(parts) == 3:
                        # append to aliasses
                        aliasses[parts[1]] = parts[2]
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
            abscissa_scale = AbscissaScale(abscissa_parts[2]) if len(abscissa_parts) >= 3 else AbscissaScale.LINEAR
            # check file contains vectors with complex numbers
            if complex:
                # complex files store the abscissa (index 0) as float64 and the remaining variables as complex128 in a structured array
                row_dtype = np.dtype([("abscissa", "<f8"), ("data", "<c16", num_variables - 1)])
                # parse binary data into a structured array with separate fields for abscissa and data variables; the abscissa is stored in a separate field to allow it to be read as float64 while the data variables are stored as complex128
                matrix = np.frombuffer(data, dtype=row_dtype, offset=binary_offset)
                # abscissa definition
                abscissa_definition = variable_definitions[0]
                # abscissa variable expression
                abscissa_expression = Expression(abscissa_definition[1], matrix["abscissa"], abscissa_definition[2].value.unit, source=None)
                # all variable expressions
                variables = [abscissa_expression] + [Expression(name, matrix["data"][:, idx - 1], var_type.value.unit, source=None) for idx, name, var_type in variable_definitions[1:]]
            else:
                # real files store all variables as float64 in a uniform layout; parse binary data into a 2D array with shape (num_points, num_variables)
                flat = np.frombuffer(data, dtype="<f8", offset=binary_offset)
                # reshape flat array into a 2D array with shape (num_points, num_variables); the data is stored in row-major order, so each row corresponds to a point and each column corresponds to a variable
                matrix = flat.reshape(num_points, num_variables)
                # extract variables
                variables = [Expression(name, matrix[:, idx], var_type.value.unit, source=None) for idx, name, var_type in variable_definitions]
            # create expression manager
            expression_manager = ExpressionManager(variables)
            # process aliasses
            if len(aliasses) > 0:
                # loop aliasses
                for alias_name, alias_expression in aliasses.items():
                    try:
                        # evaluate expression using the variables we have so far; this allows aliasses to reference previously-defined aliasses as long as there are no circular references
                        expression_manager.evaluate(alias_expression)
                    except Exception as ex:
                        # log error but continue processing other aliasses
                        logger.error("Failed to evaluate expression '%s': %s", alias_name, ex)
            # step information
            steps, abscissa = _process_step(variables[0], num_points) if stepped else (1, variables[0])
            # process scale (x axis)
            abscissa = _process_scale(abscissa, abscissa_scale)
            # create QRawFile instance with parsed header, variables, and binary data; pass the mmap so it stays alive for the lifetime of the QRawFile — Variable arrays are views into it
            return QRawFile(filename=path, title=header.get("Title", ""), date=header.get("Date", ""), plotname=header.get("Plotname", ""), complex=complex, steps=steps, abscissa=abscissa, abscissa_scale=abscissa_scale, command=header.get("Command", ""), plot_suggestion=header.get("Plot Suggestion(s)", ""), expression_manager=expression_manager, _mmap=data)
        finally:
            # log information
            logger.info("Finished loading QRAW file: %s, latency: %f seconds", path, time.perf_counter() - start_time)
