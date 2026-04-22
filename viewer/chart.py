import logging

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGraphs import QAbstractAxis, QLineSeries
from PySide6.QtQuick import QQuickItem

from .decimation_algorithm import DecimationAlgorithm, decimate_xy
from .expression import Expression
from .expression_manager import ExpressionManager
from .qraw_file import StepInformation

logger = logging.getLogger(__name__)

# default decimation algorithm used when adding series to charts
_DECIMATION_ALGORITHM = DecimationAlgorithm.M4

# series colors
_SERIES_COLOR_PALETTE = [
    "#f77f00",  # orange
    "#3a86ff",  # blue
    "#ffdd00",  # yellow
    "#9b5de5",  # indigo
    "#00b4d8",  # cyan
    "#ff8fa3",  # pink
    "#80ff72",  # green
    "#e040fb",  # purple
    "#ff4365",  # red
    "#00f5d4",  # teal
    "#f4a261",  # apricot
    "#8ac926",  # lime
    "#4cc9f0",  # sky cyan
    "#bbdefb"   # pale blue
]


class Chart:

    def __init__(self, component: QQuickItem, char_type: str, expression_manager: ExpressionManager, abscissa: Expression, step_information: StepInformation, decimate_target: int):
        # store component
        self._component = component
        # store chart type
        self._chart_type = char_type
        # store expression manager
        self._expression_manager = expression_manager
        # store variables
        self._abscissa = abscissa
        # current zoom window (x: in abscissa index, y: in ordinate percentages)
        self._zoom_window = (0, 0.0, 1, 1.0)
        # steps
        self._step_information = step_information
        self._selected_steps: set[int] = set(range(self._step_information.length))
        # store decimation target for later use when adding series
        self._decimate_target = decimate_target
        # track active series
        self._series: dict[str, tuple[Expression, dict[Expression, tuple[QAbstractAxis, dict[int, QLineSeries], float, float, str]]]] = {}
        # axis tracking for measurement types, e.g. {"V": <QAbstractAxis>, "I": <QAbstractAxis>}
        self._y_axes: dict[str, QAbstractAxis] = {}
        self._y_axes_ref_counts: dict[QAbstractAxis, int] = {}
        # axis ranges
        self._axis_ranges: dict[QAbstractAxis, tuple[float, float]] = {}
        # y axes instances
        self._left_y_axis_1: QAbstractAxis | None = None
        self._right_y_axis_1: QAbstractAxis | None = None
        self._left_y_axis_2: QAbstractAxis | None = None
        self._right_y_axis_2: QAbstractAxis | None = None
        # next color index for new series
        self._next_color_index = 0

    @property
    def expressions(self) -> list[Expression]:
        return [expression for expression, _ in self._series.values()]

    @property
    def abscissa(self) -> Expression:
        return self._abscissa

    @property
    def selected_steps(self) -> set[int]:
        return self._selected_steps

    @selected_steps.setter
    def selected_steps(self, selected_steps: set[int]) -> None:
        # check selection changed
        if selected_steps == self._selected_steps:
            return
        # update selected steps
        self._selected_steps = selected_steps
        # force step processing, adding/removing series as needed based on the new step selection and the current expressions plotted in the chart
        self.plot_series(self.expressions)

    def render(self, abscissa_label: str, abscissa_scale: str, initial_expressions: set[Expression]):
        # initialize chart component
        self._component.initialize(abscissa_label, self._abscissa.unit, abscissa_scale, self._step_information.abscissa_from_value, self._step_information.abscissa_to_value)
        # render all expressions as series
        self.plot_series(initial_expressions)
        # auto range axes based on the added series
        self.auto_range()

    def plot_series(self, expressions: set[Expression]):
        # series to render and remove from the chart
        series_to_render: list[tuple[str, str, list[QLineSeries]]] = []
        series_to_remove: list[tuple[str | None, list[QLineSeries]]] = []
        # labels to remove from the chart
        labels_to_remove: list[str] = []
        # axis to remove, prevent GC until Qt finishes its async processing of the series removals that triggered these axis removals
        axes_to_remove: list[QAbstractAxis] = []
        # loop existing series to find those that need to be removed (those whose expression is not in the new expressions list)
        for label, (expression, ordinate_series) in self._series.items():
            # check expression should be removed
            if expression not in expressions:
                # enqueue series for removal
                for ordinate_variant, (axis, rendered_series, _, _, _) in ordinate_series.items():
                    # release axis if no longer in use
                    if self._release_y_axis(axis):
                        axes_to_remove.append(axis)
                    # log information
                    logger.debug("Removing series for expression [%s] from chart, steps: %s", ordinate_variant.name, list(rendered_series.keys()))
                    # append to list for later removal from chart
                    series_to_remove.append([ordinate_variant.name, list(rendered_series.values())])
                # remove from tracked series so we don't try to update it later
                labels_to_remove.append(label)
        # update dictionary outside loop
        for label in labels_to_remove:
            del self._series[label]
        # loop expressions that should be plotted
        for ordinate in expressions:
            # lookup ordinate in series
            _, ordinate_series = self._series.get(ordinate.name, (ordinate, {}))
            # lookup expressions to plot for this ordinate, e.g. magnitude and phase for complex expressions when in AC chart
            for ordinate_variant in self._get_expressions_to_plot(ordinate):
                # looup ordinate variant in series
                y_axis, rendered_series, min_value, max_value, color = ordinate_series.get(ordinate_variant, (None, {}, float("inf"), float("-inf"), None))
                # loop rendered steps
                for step in list(rendered_series.keys()):
                    # check step should be removed
                    if step not in self._selected_steps:
                        # log information
                        logger.debug("Removing series for expression [%s] from chart, step: %d", ordinate_variant.name, step)
                        # append to list for later removal from chart
                        series_to_remove.append([None, [rendered_series[step]]])
                        # remove from dictionary so we don't try to update it later
                        del rendered_series[step]
                # process axis as needed
                if y_axis is None:
                    # find y axis for measurement type
                    y_axis = self._get_y_axis(ordinate_variant.unit)
                    if y_axis is None:
                        # log information
                        logger.warning(f"Cannot add series '{ordinate_variant.name}' of measurement type {ordinate_variant.unit} to chart — maximum number of Y axes reached")
                        # exit loop
                        break
                # check we need to generate a color for this expression
                if color is None:
                    # assign next color in palette
                    color = _SERIES_COLOR_PALETTE[self._next_color_index % len(_SERIES_COLOR_PALETTE)]
                    # update index
                    self._next_color_index += 1
                # ordinate series to render
                ordinate_series_to_render: list[QLineSeries] = []
                # loop steps
                for step in self._selected_steps:
                    # check step is already rendered
                    if step in rendered_series:
                        continue
                    # step slice
                    step_slice = self._step_information.abscissa_indices[step]

                    # # local zoom bounds for this step
                    # zoom_from_index, zoom_to_index = self._get_step_zoom_bounds(step)
                    # # apply zoom window to abscissa
                    # abscissa_values = self._abscissa.data[step_slice.start + zoom_from_index:step_slice.start + zoom_to_index]
                    # # ordinate variant values for this step (as contiguous array in memory), apply zoom window
                    # ordinate_values = ordinate_variant.data[step_slice.start + zoom_from_index:step_slice.start + zoom_to_index]

                    abscissa_values = self._abscissa.data[step_slice]
                    ordinate_values = ordinate_variant.data[step_slice]

                    # skip inconsistent slices to protect decimation input contracts
                    if abscissa_values.size == 0 or ordinate_values.size == 0 or abscissa_values.size != ordinate_values.size:
                        continue
                    # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
                    x_np, y_np = decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
                    # remove Inf values
                    inf_mask = np.isinf(y_np)
                    if inf_mask.any():
                        # mask for finite values
                        keep_mask = ~inf_mask
                        # update x and y with finite values only
                        x_np = x_np[keep_mask]
                        y_np = y_np[keep_mask]
                    # check all values were non-finite after filtering
                    if x_np.size == 0 or y_np.size == 0:
                        continue
                    # log information
                    logger.debug("Adding series for expression [%s], step: %d, original size: %d, decimated size: %d", ordinate_variant.name, step, abscissa_values.size, x_np.size)
                    # create series and hand buffers directly to Qt — no Python loop
                    series = QLineSeries()
                    series.setColor(color)
                    series.setWidth(2)
                    series.replaceNp(x_np, y_np)
                    series.setAxisY(y_axis)
                    # stroke style for stepped variants
                    if step > 0:
                        # change stroke style
                        series.setStrokeStyle(QLineSeries.StrokeStyle.DashLine)
                        series.setDashPattern([3, step + 1])
                    # append to lists
                    rendered_series[step] = series
                    # append to list for later addition to chart
                    ordinate_series_to_render.append(series)
                    # update min and max values
                    min_value = min(min_value, float(np.min(y_np)))
                    max_value = max(max_value, float(np.max(y_np)))
                # render new series
                series_to_render.append([ordinate_variant.name if ordinate_variant not in ordinate_series else None, color, ordinate_series_to_render])
                # store series with min and max values for later use when auto-ranging axes
                ordinate_series[ordinate_variant] = (y_axis, rendered_series, min_value, max_value, color)
            # store reference to allow removal later
            self._series[ordinate.name] = (ordinate, ordinate_series)
        # check changes are required in qml
        if len(series_to_render) > 0 or len(series_to_remove) > 0:
            # add/remove series from chart
            self._component.updateGraphsView(series_to_render, series_to_remove)
            # release stash after Qt finishes its async processing
            QTimer.singleShot(2000, lambda: (series_to_remove.clear(), axes_to_remove.clear()))

    def auto_range(self):
        # skip if no series are currently plotted
        if not self._series:
            return
        # min and max values axis index, reset them
        self._axis_ranges = {}
        # loop visible series
        for _, (_, ordinate_series) in self._series.items():
            # process series
            for y_axis, _, min_value, max_value, _ in ordinate_series.values():
                # current min and max for this variable type
                current_min, current_max = self._axis_ranges.get(y_axis, (float("inf"), float("-inf")))
                # compute Y values for this variable index
                self._axis_ranges[y_axis] = (min(current_min, min_value), max(current_max, max_value))
        # update axis ranges based on collected min and max values for each variable type
        for y_axis, (y_min, y_max) in self._axis_ranges.items():
            # range
            y_range = y_max - y_min
            # delta
            delta = 0.03 * y_range
            # set y axis range
            y_axis.setRange(y_min - delta, y_max + delta)

    def update_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float | None, y_bottom_ratio: float | None):
        # # vertical changes flag
        # vertical_changed = False
        # # check vertical zoom ratios were provided
        # if y_top_ratio is not None and y_bottom_ratio is not None:
        #     # current zoom window
        #     _, current_y_top_ratio, _, current_y_bottom_ratio = self._zoom_window
        #     # calculate new ratios based on the position of the mouse event within the chart panel and the current zoom window
        #     y_top_ratio = current_y_top_ratio + y_top_ratio * (current_y_bottom_ratio - current_y_top_ratio)
        #     y_bottom_ratio = current_y_top_ratio + y_bottom_ratio * (current_y_bottom_ratio - current_y_top_ratio)
        #     # update zoom window
        #     self._zoom_window = (self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
        #     # update flag
        #     vertical_changed = True
        # # check horizontal zoom indexes were provided
        # if abscissa_from_index >= 0 and abscissa_to_index >= 0:
        #     # update zoom window
        #     self._zoom_window = (abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
        #     # process all series to apply the new zoom window, full redraw if horizontal zoom changed
        #     self._redraw_all_series()
        #     # auto range axes based on the new zoom window
        #     return self.auto_range() if vertical_changed else None
        # # partial redraw sufficient when only vertical zoom changed
        # _, y_top_ratio, _, y_bottom_ratio = self._zoom_window
        # # update axis ranges based on collected min and max values for each variable type
        # for y_axis, (y_min, y_max) in self._axis_ranges.items():
        #     # range
        #     scale = y_max - y_min
        #     # set y axis range
        #     y_axis.setRange(y_min + y_top_ratio * scale, y_min + y_bottom_ratio * scale)
        ...

    def reset_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float | None, y_bottom_ratio: float | None):
        # # vertical changes flag
        # vertical_changed = False
        # # check vertical zoom ratios were provided
        # if y_top_ratio is not None and y_bottom_ratio is not None:
        #     # update flag
        #     vertical_changed = y_top_ratio != self._zoom_window[1] or y_bottom_ratio != self._zoom_window[3]
        #     # update zoom window
        #     self._zoom_window = (self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
        # # check horizontal zoom indexes were provided
        # if abscissa_from_index >= 0 and abscissa_to_index >= 0:
        #     # update zoom window
        #     self._zoom_window = (abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
        #     # process all series to apply the new zoom window, full redraw if horizontal zoom changed
        #     self._redraw_all_series()
        #     # auto range axes if vertical zoom also changed, otherwise just update axis ranges based on the new zoom window
        #     return self.auto_range() if vertical_changed else None
        # # check if vertical zoom changed, if not we can skip the redraw and just update the axis ranges based on the new zoom window
        # if vertical_changed:
        #     # log information
        #     logger.debug("Vertical zoom changed, updating axis ranges based on new zoom window")
        #     # partial redraw sufficient when only vertical zoom changed
        #     _, y_top_ratio, _, y_bottom_ratio = self._zoom_window
        #     # update axis ranges based on collected min and max values for each variable type
        #     for y_axis, (y_min, y_max) in self._axis_ranges.items():
        #         # set y axis range
        #         y_axis.setRange(y_min, y_max)
        ...

    def clear(self):
        # Qt enqueues the visual removal of series asynchronously. Python owns the QLineSeries, so we must NOT let Python GC them until Qt has finished processing the removal queue.
        old_series = self._series
        old_y_axes = self._y_axes
        # reset internal state
        self._series = {}
        self._y_axes = {}
        self._y_axes_ref_counts = {}
        # reset color index for new series
        self._next_color_index = 0
        # reset zoom window (vertical axes only, keep horizontal range)
        self._zoom_window = (self._zoom_window[0], 0.0, self._zoom_window[2], 1.0)
        # enqueue Qt-side removal
        self._component.removeAllSeries()
        # remove axis references
        self._left_y_axis_1 = None
        self._right_y_axis_1 = None
        self._left_y_axis_2 = None
        self._right_y_axis_2 = None
        # release stash after Qt finishes its async processing
        QTimer.singleShot(1000, lambda: (old_series.clear(), old_y_axes.clear()))

    def sample_at(self, x_ratio: float) -> list[tuple[str, str, list[float]]]:
        # check series are plotted
        if not self._series:
            return []
        # clamped horizontal ratio in visible window
        x_ratio = max(0.0, min(1.0, x_ratio))
        # current horizontal zoom bounds
        from_index, to_index = self.zoom_abscissa_bounds()
        # reference step for x mapping, using the first selected step when available
        reference_step = min(self._selected_steps) if self._selected_steps else 0
        # reference step slice
        reference_slice = self._step_information.abscissa_indices[reference_step]
        # visible abscissa window for the reference step
        window = self._abscissa.data[reference_slice.start + from_index:reference_slice.start + to_index]
        # check visible window is empty
        if window.size == 0:
            return []
        # collect one (name, unit, value) tuple per plotted variant (magnitude/phase counted separately)
        result: list[tuple[str, str, list[float]]] = []
        # loop series
        for _, (_, ordinate_series) in self._series.items():
            # loop variants for this series (magnitude/phase)
            for ordinate_variant, (_, rendered_series, _, _, _) in ordinate_series.items():
                # values (per step)
                values: list[float] = []
                # # step and series for this step
                # for step, _ in rendered_series.items():
                #     # step slice
                #     step_slice = self._step_information.abscissa_indices[step]
                #     # fallback to raw value at nearest original sample index (this is not correct, idx is for decimated points)
                #     values.append(float(ordinate_variant.data[step_slice][idx]))
                # append to result (name, unit, value)
                result.append((ordinate_variant.name, ordinate_variant.unit, values))
        # exit
        return result

    def zoom_abscissa_bounds(self) -> tuple[int, int]:
        # current abscissa bounds in sample indexes
        return self._zoom_window[0], self._zoom_window[2]

    def _get_step_zoom_bounds(self, step: int) -> tuple[int, int]:
        # step length for this step
        step_length = self._step_information.step_length(step)
        # keep at least two points when possible for stable zoom and FFT windows
        if step_length <= 1:
            return 0, step_length
        # clamp zoom start to step bounds
        from_index = max(0, min(self._zoom_window[0], step_length - 2))
        # clamp zoom end to step bounds and enforce a minimum window size of two points
        to_index = max(from_index + 2, min(self._zoom_window[2], step_length))
        # return local bounds
        return from_index, to_index

    def sample_index_at_ratio(self, x_ratio: float) -> int:
        # x zoom window indexes
        from_index, to_index = self.zoom_abscissa_bounds()
        # reference step for x mapping, using the first selected step when available
        reference_step = min(self._selected_steps) if self._selected_steps else 0
        # clamp bounds to the reference step
        from_index, to_index = self._get_step_zoom_bounds(reference_step)
        # reference slice
        reference_slice = self._step_information.abscissa_indices[reference_step]
        # clamp ratio to visible horizontal span
        x_ratio = max(0.0, min(1.0, x_ratio))
        # window length
        window_len = to_index - from_index
        # check visible window has a single point
        if window_len <= 1:
            # return only sample index in window
            return from_index
        # visible abscissa window which can be non-uniformly spaced
        window = self._abscissa.data[reference_slice.start + from_index:reference_slice.start + to_index]
        # target x value on the visible axis for the given cursor ratio
        target = float(window[0] + x_ratio * (window[-1] - window[0]))
        # check ascending abscissa ordering
        if window[0] <= window[-1]:
            # insertion index in visible window
            insert_at = int(np.searchsorted(window, target, side="left"))
            # check target is left of the first visible sample
            if insert_at <= 0:
                # clamp to left bound
                return from_index
            # check target is right of the last visible sample
            if insert_at >= window_len:
                # clamp to right bound
                return to_index - 1
            # nearest neighbors around insertion point
            left_idx = insert_at - 1
            right_idx = insert_at
            # distance to left neighbor
            left_dist = abs(float(window[left_idx]) - target)
            # distance to right neighbor
            right_dist = abs(float(window[right_idx]) - target)
            # select the closest index and break ties to the left
            return from_index + (left_idx if left_dist <= right_dist else right_idx)
        # descending abscissa search on reversed view
        reversed_window = window[::-1]
        # insertion index in reversed visible window
        insert_at = int(np.searchsorted(reversed_window, target, side="left"))
        # check target is left of the first visible sample in descending order
        if insert_at <= 0:
            # clamp to right bound in original ordering
            return to_index - 1
        # check target is right of the last visible sample in descending order
        if insert_at >= window_len:
            # clamp to left bound in original ordering
            return from_index
        # map insertion point neighbors back to original ordering
        right_idx = window_len - insert_at
        left_idx = right_idx - 1
        # distance to left neighbor
        left_dist = abs(float(window[left_idx]) - target)
        # distance to right neighbor
        right_dist = abs(float(window[right_idx]) - target)
        # select the closest index and break ties to the left
        return from_index + (left_idx if left_dist <= right_dist else right_idx)

    def sample_abscissa_value_at_ratio(self, x_ratio: float) -> float:
        # reference step for x mapping, using the first selected step when available
        reference_step = min(self._selected_steps) if self._selected_steps else 0
        # reference step slice
        reference_slice = self._step_information.abscissa_indices[reference_step]
        # nearest sample index within the current zoom window for the reference step
        index = self.sample_index_at_ratio(x_ratio)
        # return stored abscissa value
        return float(self._abscissa.data[reference_slice.start + index])

    def _get_expressions_to_plot(self, expression: Expression) -> list[Expression]:
        # check we can plot expression as is
        if not expression.complex:
            return [expression]
        # check chart type
        if self._chart_type == "AC":
            # magnitude
            magnitude_expression = self._expression_manager.evaluate(f"db({expression.name})")
            if not magnitude_expression:
                return []
            # phase
            phase_expression = self._expression_manager.evaluate(f"phase({expression.name})")
            if not phase_expression:
                return []
            # exit
            return [magnitude_expression, phase_expression]
        # complex expressions are only renderable in AC charts; skip with a warning for other chart types
        logger.warning("skipping complex expression '%s': not supported for chart type '%s'", expression.name, self._chart_type)
        # exit
        return []

    def _redraw_all_series(self):
        # # x0 and x1
        # abscissa_left_value: float | None = None
        # abscissa_right_value: float | None = None
        # try:
        #     # loop existing series
        #     for _, (_, ordinate_series) in self._series.items():
        #         # loop series data (actual data visible in chart)
        #         for ordinate_variant, (y_axis, rendered_series, _, _, color) in ordinate_series.items():
        #             # min and max value recalculation for the new zoom window
        #             min_value = float("inf")
        #             max_value = float("-inf")
        #             # loop steps
        #             for step, series in rendered_series.items():
        #                 # step slice
        #                 step_slice = self._step_information.abscissa_indices[step]
        #                 # local zoom bounds for this step
        #                 zoom_from_index, zoom_to_index = self._get_step_zoom_bounds(step)
        #                 # abscissa values
        #                 abscissa_values = self._abscissa.data[step_slice.start + zoom_from_index:step_slice.start + zoom_to_index]
        #                 # ordinate variant values for this step & zoom window
        #                 ordinate_values = ordinate_variant.data[step_slice.start + zoom_from_index:step_slice.start + zoom_to_index]
        #                 # skip inconsistent slices to protect decimation input contracts
        #                 if abscissa_values.size == 0 or ordinate_values.size == 0 or abscissa_values.size != ordinate_values.size:
        #                     continue
        #                 # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
        #                 x_np, y_np = decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
        #                 # remove Inf values
        #                 inf_mask = np.isinf(y_np)
        #                 if inf_mask.any():
        #                     # mask for finite values
        #                     keep_mask = ~inf_mask
        #                     # update x and y with finite values only
        #                     x_np = x_np[keep_mask]
        #                     y_np = y_np[keep_mask]
        #                 # check all values were non-finite after filtering
        #                 if x_np.size == 0 or y_np.size == 0:
        #                     continue
        #                 # update series with decimated data
        #                 series.replaceNp(x_np, y_np)
        #                 # update min and max values
        #                 min_value = min(min_value, float(np.min(y_np)))
        #                 max_value = max(max_value, float(np.max(y_np)))
        #                 # update x axis left and right values based on the new zoom window
        #                 if abscissa_left_value is None or abscissa_right_value is None:
        #                     # initialize values for the first series processed
        #                     abscissa_left_value = float(abscissa_values[0])
        #                     abscissa_right_value = float(abscissa_values[-1])
        #                 elif abscissa_left_value < abscissa_right_value:
        #                     # ascending abscissa, update left and right values as needed
        #                     abscissa_left_value = min(abscissa_left_value, float(abscissa_values[0]))
        #                     abscissa_right_value = max(abscissa_right_value, float(abscissa_values[-1]))
        #                 else:
        #                     # descending abscissa, update left and right values as needed
        #                     abscissa_left_value = max(abscissa_left_value, float(abscissa_values[0]))
        #                     abscissa_right_value = min(abscissa_right_value, float(abscissa_values[-1]))
        #             # calculate scale for Y axis
        #             scale = max(abs(max_value), abs(min_value))
        #             # Y axis range
        #             y_range = max_value - min_value
        #             if y_range <= scale * 1e-9:
        #                 y_range = abs(max_value) * 0.01 if scale != 0 else 1.0
        #             # protect against very small ranges
        #             delta = max(0.01 * y_range, 1e-3)
        #             # update dictionary entry
        #             ordinate_series[ordinate_variant] = (y_axis, rendered_series, min_value - delta, max_value + delta, color)
        # finally:
        #     # resize abscissa axis
        #     if abscissa_left_value is not None and abscissa_right_value is not None:
        #         self._component.resizeAbscissa(abscissa_left_value, abscissa_right_value)
        ...

    def _get_y_axis(self, unit: str) -> QAbstractAxis | None:
        # existing axis for measurement type
        axis = self._y_axes.get(unit)
        if axis is not None:
            # increase reference count for this axis
            self._y_axes_ref_counts[axis] += 1
            # use axis
            return axis
        # log information
        logger.debug("Creating Y axis for measurement type: %s", unit or "<no unit>")
        # left (main)
        if self._left_y_axis_1 is None:
            # create axis
            self._left_y_axis_1 = self._component.createYAxis(Qt.AlignmentFlag.AlignLeft, unit)
            # register axis
            self._y_axes[unit] = self._left_y_axis_1
            self._y_axes_ref_counts[self._left_y_axis_1] = 1
            # use axis
            return self._left_y_axis_1
        # right (main)
        if self._right_y_axis_1 is None:
            # create axis
            self._right_y_axis_1 = self._component.createYAxis(Qt.AlignmentFlag.AlignRight, unit)
            # register axis
            self._y_axes[unit] = self._right_y_axis_1
            self._y_axes_ref_counts[self._right_y_axis_1] = 1
            # use axis
            return self._right_y_axis_1
        # left (secondary)
        if self._left_y_axis_2 is None:
            # create axis
            self._left_y_axis_2 = self._component.createYAxis(Qt.AlignmentFlag.AlignLeft, unit)
            # register axis
            self._y_axes[unit] = self._left_y_axis_2
            self._y_axes_ref_counts[self._left_y_axis_2] = 1
            # use axis
            return self._left_y_axis_2
        # right (secondary)
        if self._right_y_axis_2 is None:
            # create axis
            self._right_y_axis_2 = self._component.createYAxis(Qt.AlignmentFlag.AlignRight, unit)
            # register axis
            self._y_axes[unit] = self._right_y_axis_2
            self._y_axes_ref_counts[self._right_y_axis_2] = 1
            # use axis
            return self._right_y_axis_2
        # no more axes available
        return None

    def _release_y_axis(self, axis: QAbstractAxis) -> bool:
        # decrease reference count for this axis
        self._y_axes_ref_counts[axis] -= 1
        # check if axis is now unused and can be released
        if self._y_axes_ref_counts[axis] == 0:
            # unit
            unit = axis.property("yUnit")
            # log information
            logger.debug("Releasing Y axis for measurement type: %s", unit or "<no unit>")
            # remove from tracked axes
            del self._y_axes_ref_counts[axis]
            del self._y_axes[unit]
            # release the internal reference
            if axis == self._left_y_axis_1:
                self._left_y_axis_1 = None
            elif axis == self._right_y_axis_1:
                self._right_y_axis_1 = None
            elif axis == self._left_y_axis_2:
                self._left_y_axis_2 = None
            elif axis == self._right_y_axis_2:
                self._right_y_axis_2 = None
            # remove from chart
            return True
        # exit
        return False
