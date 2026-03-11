import logging

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGraphs import QAbstractAxis, QLineSeries
from PySide6.QtQuick import QQuickItem

from .decimation_algorithm import DecimationAlgorithm, decimate_xy
from .expression import Expression
from .expression_manager import ExpressionManager

logger = logging.getLogger(__name__)

# default decimation algorithm used when adding series to charts
_DECIMATION_ALGORITHM = DecimationAlgorithm.M4


class Chart:

    def __init__(self, component: QQuickItem, expression_manager: ExpressionManager, abscissa: Expression, abscissa_from_index: int, abscissa_to_index: int, steps: int, decimate_target: int):
        # store component
        self._component = component
        # store expression manager
        self._expression_manager = expression_manager
        # store variables
        self._abscissa = abscissa
        self._expressions: list[Expression] = []
        # current zoom window (x: in abscissa index, y: in ordinate percentages)
        self._zoom_window = (abscissa_from_index, 0.0, abscissa_to_index, 1.0)
        # steps
        self._steps = steps
        self._step_points = len(abscissa.data)
        # store decimation target for later use when adding series
        self._decimate_target = decimate_target
        # track active series
        self._series: dict[str, tuple[Expression, list[tuple[Expression, QAbstractAxis, list[QLineSeries], float, float]]]] = {}
        # track Y axis usage
        self._y_axes: dict[str, QAbstractAxis] = {}
        # axis ranges
        self._axis_ranges: dict[QAbstractAxis, tuple[float, float]] = {}

    @property
    def expressions(self) -> list[Expression]:
        # return copy of expressions list to prevent external mutation of the chart's internal state, which would cause inconsistencies between the plotted series and the expressions list
        return self._expressions[:]

    @property
    def abscissa(self) -> Expression:
        return self._abscissa

    def render(self, abscissa_label: str, abscissa_scale: str, initial_expressions: set[Expression]):
        # x0 and x1
        abscissa_left_value = float(self._abscissa.data[self._zoom_window[0]])
        abscissa_right_value = float(self._abscissa.data[self._zoom_window[2] - 1])
        # initialize chart component
        self._component.initialize(abscissa_label, self._abscissa.unit, abscissa_scale, abscissa_left_value, abscissa_right_value)
        # render all expressions as series
        self.plot_series(initial_expressions)
        # auto range axes based on the added series
        self.auto_range()

    def plot_series(self, expressions: set[Expression]):
        # apply zoom window to abscissa — shared across all ordinate steps, will be paired with y below
        abscissa_values = self._abscissa.data[self._zoom_window[0]:self._zoom_window[2]]
        # series to render and remove from the chart
        series_to_render: list[tuple[str, list[QLineSeries]]] = []
        series_to_remove: list[tuple[str, list[QLineSeries]]] = []
        # labels to remove from the chart
        labels_to_remove: list[str] = []
        # loop existing series to find those that need to be removed (those whose expression is not in the new expressions list)
        for label, (expression, ordinate_series) in self._series.items():
            # check expression should be removed
            if expression not in expressions:
                # log information
                logger.debug("Removing series for expression '%s' from chart", expression.name)
                # remove from expressions
                self._expressions.remove(expression)
                # enqueue series for removal
                for ordinate_variant, _, series_list, _, _ in ordinate_series:
                    series_to_remove.append([ordinate_variant.name, series_list])
                # remove from tracked series so we don't try to update it later
                labels_to_remove.append(label)
        # update dictionary outside loop
        for label in labels_to_remove:
            del self._series[label]
        # loop expressions that should be plotted
        for ordinate in expressions:
            # skip if a series with this label is already plotted
            if ordinate.name in self._series:
                continue
            # store expression
            self._expressions.append(ordinate)
            # ordinate series
            ordinate_series: list[tuple[Expression, QAbstractAxis, list[QLineSeries], float, float]] = []
            # check ordinate represents a complex number, if this is the case split measurement into magnitude and phase
            for ordinate_variant in self._get_expressions_to_plot(ordinate):
                # find y axis for measurement type
                y_axis = self._get_y_axis(ordinate_variant.unit)
                if y_axis is None:
                    # log information
                    logger.warning(f"Cannot add series '{ordinate_variant.name}' of measurement type {ordinate_variant.unit} to chart — maximum number of Y axes reached")
                    # exit loop
                    break
                # ordinate series
                ordinate_variant_series: list[QLineSeries] = []
                # min and max values for this variable variant (across all steps)
                min_value = float("inf")
                max_value = float("-inf")
                # loop steps
                for step in range(self._steps):
                    # ordinate variant values for this step (as contiguous array in memory), apply zoom window
                    ordinate_values = ordinate_variant.data[step * self._step_points: (step + 1) * self._step_points][self._zoom_window[0]:self._zoom_window[2]]
                    # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
                    x_np, y_np = decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
                    # remove NaN, Inf and -Inf values which can cause issues for chart when plotting
                    finite_mask = np.isfinite(y_np)
                    # update x and y with finite values only
                    x_np = x_np[finite_mask]
                    y_np = y_np[finite_mask]
                    # create series and hand buffers directly to Qt — no Python loop
                    series = QLineSeries()
                    series.setWidth(1)
                    series.replaceNp(x_np, y_np)
                    series.setAxisY(y_axis)
                    # append to lists
                    ordinate_variant_series.append(series)
                    # update min and max values
                    min_value = min(min_value, float(np.min(y_np)))
                    max_value = max(max_value, float(np.max(y_np)))
                # append to lists
                series_to_render.append([ordinate_variant.name, ordinate_variant_series])
                # calculate scale for Y axis
                scale = max(abs(max_value), abs(min_value))
                # Y axis range
                y_range = max_value - min_value
                if y_range <= scale * 1e-9:
                    y_range = abs(max_value) * 0.01 if scale != 0 else 1.0
                # store series with min and max values for later use when auto-ranging axes
                ordinate_series.append((ordinate_variant, y_axis, ordinate_variant_series, min_value - 0.01 * y_range, max_value + 0.01 * y_range))
            # store reference to allow removal later
            self._series[ordinate.name] = (ordinate, ordinate_series)
        # add/remove series from chart
        self._component.plotSeries(series_to_render, series_to_remove)
        # release stash after Qt finishes its async processing
        QTimer.singleShot(2000, lambda: (series_to_remove.clear()))

    def auto_range(self):
        # skip if no series are currently plotted
        if not self._series:
            return
        # min and max values axis index, reset them
        self._axis_ranges = {}
        # loop visible series
        for _, (_, ordinate_series) in self._series.items():
            # process series
            for _, y_axis, _, min_value, max_value in ordinate_series:
                # current min and max for this variable type
                current_min, current_max = self._axis_ranges.get(y_axis, (float("inf"), float("-inf")))
                # compute Y values for this variable index
                self._axis_ranges[y_axis] = (min(current_min, min_value), max(current_max, max_value))
        # update axis ranges based on collected min and max values for each variable type
        for y_axis, (y_min, y_max) in self._axis_ranges.items():
            # set y axis range
            y_axis.setRange(y_min, y_max)

    def update_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float | None, y_bottom_ratio: float | None):
        # vertical changes flag
        vertical_changed = False
        # check vertical zoom ratios were provided
        if y_top_ratio is not None and y_bottom_ratio is not None:
            # current zoom window
            _, current_y_top_ratio, _, current_y_bottom_ratio = self._zoom_window
            # calculate new ratios based on the position of the mouse event within the chart panel and the current zoom window
            y_top_ratio = current_y_top_ratio + y_top_ratio * (current_y_bottom_ratio - current_y_top_ratio)
            y_bottom_ratio = current_y_top_ratio + y_bottom_ratio * (current_y_bottom_ratio - current_y_top_ratio)
            # update zoom window
            self._zoom_window = (self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
            # update flag
            vertical_changed = True
        # check horizontal zoom indexes were provided
        if abscissa_from_index >= 0 and abscissa_to_index >= 0:
            # update zoom window
            self._zoom_window = (abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
            # process all series to apply the new zoom window, full redraw if horizontal zoom changed
            self._redraw_all_series()
            # auto range axes based on the new zoom window
            return self.auto_range() if vertical_changed else None
        # partial redraw sufficient when only vertical zoom changed
        _, y_top_ratio, _, y_bottom_ratio = self._zoom_window
        # update axis ranges based on collected min and max values for each variable type
        for y_axis, (y_min, y_max) in self._axis_ranges.items():
            # range
            scale = y_max - y_min
            # set y axis range
            y_axis.setRange(y_min + y_top_ratio * scale, y_min + y_bottom_ratio * scale)

    def reset_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float | None, y_bottom_ratio: float | None):
        # vertical changes flag
        vertical_changed = False
        # check vertical zoom ratios were provided
        if y_top_ratio is not None and y_bottom_ratio is not None:
            # update flag
            vertical_changed = y_top_ratio != self._zoom_window[1] or y_bottom_ratio != self._zoom_window[3]
            # update zoom window
            self._zoom_window = (self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
        # check horizontal zoom indexes were provided
        if abscissa_from_index >= 0 and abscissa_to_index >= 0:
            # update zoom window
            self._zoom_window = (abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
            # process all series to apply the new zoom window, full redraw if horizontal zoom changed
            self._redraw_all_series()
            # auto range axes if vertical zoom also changed, otherwise just update axis ranges based on the new zoom window
            return self.auto_range() if vertical_changed else None
        # check if vertical zoom changed, if not we can skip the redraw and just update the axis ranges based on the new zoom window
        if vertical_changed:
            # log information
            logger.debug("Vertical zoom changed, updating axis ranges based on new zoom window")
            # partial redraw sufficient when only vertical zoom changed
            _, y_top_ratio, _, y_bottom_ratio = self._zoom_window
            # update axis ranges based on collected min and max values for each variable type
            for y_axis, (y_min, y_max) in self._axis_ranges.items():
                # set y axis range
                y_axis.setRange(y_min, y_max)

    def clear(self):
        # Qt enqueues the visual removal of series asynchronously. Python owns the QLineSeries, so we must NOT let Python GC them until Qt has finished processing the removal queue.
        old_series = self._series
        old_axes = self._y_axes
        old_expressions = self._expressions
        # reset internal state
        self._series = {}
        self._y_axes = {}
        self._expressions = []
        # reset zoom window (vertical axes only, keep horizontal range)
        self._zoom_window = (self._zoom_window[0], 0.0, self._zoom_window[2], 1.0)
        # enqueue Qt-side removal
        self._component.removeAllSeries()
        # release stash after Qt finishes its async processing
        QTimer.singleShot(1000, lambda: (old_series.clear(), old_axes.clear(), old_expressions.clear()))

    def sample_at(self, x_ratio: float) -> list[tuple[str, str, list[float]]]:
        # check series are plotted
        if not self._series:
            return []
        # x zoom window indexes
        from_index, _, to_index, _ = self._zoom_window
        # compute the nearest sample index within the current zoom window
        idx = max(from_index, min(to_index - 1, int(round(from_index + x_ratio * (to_index - from_index)))))
        # collect one (name, unit, value) tuple per plotted variant (magnitude/phase counted separately)
        result: list[tuple[str, str, list[float]]] = []
        # loop series
        for _, (_, ordinate_series) in self._series.items():
            # loop variants for this series (magnitude/phase)
            for ordinate_variant, _, series_list, _, _ in ordinate_series:
                # values (per step)
                values: list[float] = []
                # loop series list (steps)
                for step, _ in enumerate(series_list):
                    # value for step at index
                    values.append(float(ordinate_variant.data[step * self._step_points + idx]))
                # append to result (name, unit, value)
                result.append((ordinate_variant.name, ordinate_variant.unit, values))
        # exit
        return result

    def _get_expressions_to_plot(self, expression: Expression) -> list[Expression]:
        # check we can plot expression as is
        if not expression.complex:
            return [expression]
        # magniture
        magnitude_expression = self._expression_manager.evaluate(f"abs({expression.name})")
        if not magnitude_expression:
            return []
        # phase
        phase_expression = self._expression_manager.evaluate(f"angle({expression.name})")
        if not phase_expression:
            return []
        # exit
        return [magnitude_expression, phase_expression]

    def _get_y_axis(self, unit: str) -> QAbstractAxis | None:
        # existing axis for measurement type
        axis = self._y_axes.get(unit)
        if axis is not None:
            return axis
        # log information
        logger.debug("Reserving Y axis [%d] for measurement type: %s", len(self._y_axes), unit)
        # number of y axis in use
        counter = len(self._y_axes)
        # check we have reached the maximum number of Y axes allowed by this chart type
        if counter == 4:
            return None
        # create axis
        axis = self._component.createYAxis(f"Y Axis {counter + 1}", unit)
        # reserve axis for this measurement type
        self._y_axes[unit] = axis
        # exit
        return axis

    def _redraw_all_series(self):
        # abscissa values — shared across all ordinate steps, will be paired with y below
        abscissa_values = self._abscissa.data[self._zoom_window[0]:self._zoom_window[2]]
        # x0 and x1
        abscissa_min = float(abscissa_values[0])
        abscissa_max = float(abscissa_values[-1])
        try:
            # loop existing series
            for _, (_, ordinate_series) in self._series.items():
                # recalculated min and max values for this variable across all steps based on the new zoom window
                new_ordinate_series = []
                # loop series data (actual data visible in chart)
                for ordinate_variant, y_axis, series_list, _, _ in ordinate_series:
                    # min and max value recalculation for the new zoom window
                    min_value = float("inf")
                    max_value = float("-inf")
                    # loop steps
                    for step in range(self._steps):
                        # ordinate variant values for this step
                        ordinate_values = ordinate_variant.data[step * self._step_points: (step + 1) * self._step_points][self._zoom_window[0]:self._zoom_window[2]]
                        # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
                        x_np, y_np = decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
                        # remove NaN, Inf and -Inf values which can cause issues for chart when plotting
                        finite_mask = np.isfinite(y_np)
                        # update x and y with finite values only
                        x_np = x_np[finite_mask]
                        y_np = y_np[finite_mask]
                        # update series with decimated data
                        series_list[step].replaceNp(x_np, y_np)
                        # update min and max values
                        min_value = min(min_value, float(np.min(y_np)))
                        max_value = max(max_value, float(np.max(y_np)))
                    # calculate scale for Y axis
                    scale = max(abs(max_value), abs(min_value))
                    # Y axis range
                    y_range = max_value - min_value
                    if y_range <= scale * 1e-9:
                        y_range = abs(max_value) * 0.01 if scale != 0 else 1.0
                    # append to list for later update of the series data and axis ranges after the loop
                    new_ordinate_series.append((ordinate_variant, y_axis, series_list, min_value - 0.01 * y_range, max_value + 0.01 * y_range))
                # replace the list contents in-place
                ordinate_series[:] = new_ordinate_series
        finally:
            # resize abscissa axis
            self._component.resizeAbscissa(abscissa_min, abscissa_max)
