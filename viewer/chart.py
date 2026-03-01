import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGraphs import QAbstractAxis, QLineSeries
from PySide6.QtQuick import QQuickItem

from .decimation_algorithm import DecimationAlgorithm, decimate_xy
from .variable import Variable, VariableType

logger = logging.getLogger(__name__)

# default decimation algorithm used when adding series to charts
_DECIMATION_ALGORITHM = DecimationAlgorithm.M4


class Chart:

    def __init__(self, component: QQuickItem, abscissa: Variable, abscissa_from_index: int, abscissa_to_index: int, decimate_target: int):
        # store component
        self._component = component
        # store variables
        self._abscissa = abscissa
        self._variables: list[Variable] = []
        # current zoom window (x: in abscissa index, y: in ordinate percentages)
        self._zoom_window = (abscissa_from_index, 0.0, abscissa_to_index, 1.0)
        # store decimation target for later use when adding series
        self._decimate_target = decimate_target
        # track active series
        self._series: dict[str, tuple[Variable, list[tuple[Variable, QAbstractAxis, list[QLineSeries], float, float]]]] = {}
        # track one Y axis usage
        self._y_axes: dict[VariableType, QAbstractAxis] = {}
        # axis ranges
        self._axis_ranges: dict[QAbstractAxis, tuple[float, float]] = {}

    @property
    def variables(self) -> list[Variable]:
        # return copy of variables list to prevent external mutation of the chart's internal state, which would cause inconsistencies between the plotted series and the variables list
        return self._variables[:]

    def render(self, abscissa_label: str, abscissa_scale: str, initial_variables: set[Variable]):
        # x0 and x1
        abscissa_left_value = float(self._abscissa.values[self._zoom_window[0]])
        abscissa_right_value = float(self._abscissa.values[self._zoom_window[2] - 1])
        # initialize chart component
        self._component.initialize(abscissa_label, self._abscissa.type.value.unit, abscissa_scale, abscissa_left_value, abscissa_right_value)
        # render all variables as series
        self.plot_series(initial_variables)
        # auto range axes based on the added series
        self.auto_range()

    def plot_series(self, variables: set[Variable]):
        # abscissa values — shared across all ordinate steps, will be paired with y below
        abscissa_values = self._abscissa.values[self._zoom_window[0]:self._zoom_window[2]]
        # series to render and remove from the chart
        series_to_render: list = []
        series_to_remove: list = []
        # labels to remove from the chart
        labels_to_remove: list[str] = []
        # loop existing series to find those that need to be removed (those whose variable is not in the new variables list)
        for label, (variable, ordinate_series) in self._series.items():
            # check expression should be removed
            if variable not in variables:
                # log information
                logger.debug("Removing series for variable '%s' from chart", variable.name)
                # remove from variables
                self._variables.remove(variable)
                # enqueue series for removal
                for ordinate_variant, _, series_list, _, _ in ordinate_series:
                    series_to_remove.append([ordinate_variant.name, series_list])
                # remove from tracked series so we don't try to update it later
                labels_to_remove.append(label)
        # update dictionary outside loop
        for label in labels_to_remove:
            del self._series[label]
        # loop variables that should be plotted
        for ordinate in variables:
            # skip if a series with this label is already plotted
            if ordinate.name in self._series:
                continue
            # store variable
            self._variables.append(ordinate)
            # ordinate series
            ordinate_series: list[tuple[Variable, QAbstractAxis, list[QLineSeries], float, float]] = []
            # check ordinate represents a complex number, if this is the case split measurement into magnitude and phase
            for ordinate_variant in [ordinate.magnitude, ordinate.phase] if ordinate.complex else [ordinate]:
                # find y axis for variable type
                y_axis = self._get_y_axis(ordinate_variant.type)
                if y_axis is None:
                    # log information
                    logger.warning(f"Cannot add series '{ordinate_variant.name}' of variable type {ordinate_variant.type.name} to chart — maximum number of Y axes reached")
                    # exit loop
                    break
                # ordinate series
                ordinate_variant_series: list[QLineSeries] = []
                # min and max values for this variable variant (across all steps)
                min_value = float("inf")
                max_value = float("-inf")
                # loop steps
                for step in range(self._abscissa.steps):
                    # ordinate variant values for this step
                    ordinate_values = ordinate_variant.step_values(step)[self._zoom_window[0]:self._zoom_window[2]]
                    # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
                    x_np, y_np = decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
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
                ordinate_series.append((ordinate_variant, y_axis, ordinate_variant_series, min_value, max_value))
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
        self._axis_ranges={}
        # loop visible series
        for _, (_, ordinate_series) in self._series.items():
            # process series
            for _, y_axis, _, min_value, max_value in ordinate_series:
                # current min and max for this variable type
                current_min, current_max=self._axis_ranges.get(y_axis, (float("inf"), float("-inf")))
                # compute Y values for this variable index
                self._axis_ranges[y_axis]=(min(current_min, min_value), max(current_max, max_value))
        # zoom ratios
        _, y_top_ratio, _, y_bottom_ratio=self._zoom_window
        # update axis ranges based on collected min and max values for each variable type
        for y_axis, (y_min, y_max) in self._axis_ranges.items():
            # set y axis range
            y_axis.setRange(y_min + y_top_ratio * (y_max - y_min), y_min + y_bottom_ratio * (y_max - y_min))

    def update_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float, y_bottom_ratio: float):
        # vertical changes flag
        vertical_changed=False
        # check vertical zoom ratios were provided
        if y_top_ratio >= 0 and y_bottom_ratio >= 0:
            # current zoom window
            _, current_y_top_ratio, _, current_y_bottom_ratio=self._zoom_window
            # calculate new ratios based on the position of the mouse event within the chart panel and the current zoom window
            y_top_ratio=current_y_top_ratio + y_top_ratio * (current_y_bottom_ratio - current_y_top_ratio)
            y_bottom_ratio=current_y_top_ratio + y_bottom_ratio * (current_y_bottom_ratio - current_y_top_ratio)
            # update zoom window
            self._zoom_window=(self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
            # update flag
            vertical_changed=True
        # check horizontal zoom indexes were provided
        if abscissa_from_index >= 0 and abscissa_to_index >= 0:
            # update zoom window
            self._zoom_window=(abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
            # process all series to apply the new zoom window, full redraw if horizontal zoom changed
            self._redraw_all_series()
            # auto range axes based on the new zoom window
            return self.auto_range() if vertical_changed else None
        # partial redraw sufficient when only vertical zoom changed
        _, y_top_ratio, _, y_bottom_ratio=self._zoom_window
        # update axis ranges based on collected min and max values for each variable type
        for y_axis, (y_min, y_max) in self._axis_ranges.items():
            # set y axis range
            y_axis.setRange(y_min + y_top_ratio * (y_max - y_min), y_min + y_bottom_ratio * (y_max - y_min))

    def reset_zoom_window(self, abscissa_from_index: int, abscissa_to_index: int, y_top_ratio: float, y_bottom_ratio: float):
        # vertical changes flag
        vertical_changed=False
        # check vertical zoom ratios were provided
        if y_top_ratio >= 0 and y_bottom_ratio >= 0:
            # update flag
            vertical_changed=y_top_ratio != self._zoom_window[1] or y_bottom_ratio != self._zoom_window[3]
            # update zoom window
            self._zoom_window=(self._zoom_window[0], y_top_ratio, self._zoom_window[2], y_bottom_ratio)
        # check horizontal zoom indexes were provided
        if abscissa_from_index >= 0 and abscissa_to_index >= 0:
            # update zoom window
            self._zoom_window=(abscissa_from_index, self._zoom_window[1], abscissa_to_index, self._zoom_window[3])
            # process all series to apply the new zoom window, full redraw if horizontal zoom changed
            self._redraw_all_series()
            # auto range axes based on the new zoom window
            return self.auto_range()
        # check if vertical zoom changed, if not we can skip the redraw and just update the axis ranges based on the new zoom window
        if vertical_changed:
            # log information
            logger.debug("Vertical zoom changed, updating axis ranges based on new zoom window")
            # partial redraw sufficient when only vertical zoom changed
            _, y_top_ratio, _, y_bottom_ratio=self._zoom_window
            # update axis ranges based on collected min and max values for each variable type
            for y_axis, (y_min, y_max) in self._axis_ranges.items():
                # set y axis range
                y_axis.setRange(y_min + y_top_ratio * (y_max - y_min), y_min + y_bottom_ratio * (y_max - y_min))

    def clear(self):
        # Qt enqueues the visual removal of series asynchronously. Python owns the QLineSeries, so we must NOT let
        # Python GC them until Qt has finished processing the removal queue.
        old_series=self._series
        old_axes=self._y_axes
        old_variables=self._variables
        # reset internal state
        self._series={}
        self._y_axes={}
        self._variables=[]
        # reset zoom window (vertical axes only, keep horizontal range)
        self._zoom_window=(self._zoom_window[0], 0.0, self._zoom_window[2], 1.0)
        # enqueue Qt-side removal
        self._component.removeAllSeries()
        # release stash after Qt finishes its async processing
        QTimer.singleShot(1000, lambda: (old_series.clear(), old_axes.clear(), old_variables.clear()))

    def _get_y_axis(self, variable_type: VariableType) -> QAbstractAxis | None:
        # existing axis for variable type
        axis=self._y_axes.get(variable_type)
        if axis is not None:
            return axis
        # log information
        logger.debug("Reserving Y axis [%d] for variable type: %s", len(self._y_axes), variable_type.name)
        # number of y axis in use
        counter=len(self._y_axes)
        # check we have reached the maximum number of Y axes allowed by this chart type
        if counter == 4:
            return None
        # create axis
        axis=self._component.createYAxis(f"Y Axis {counter + 1}", variable_type.value.unit)
        # reserve axis for this variable type
        self._y_axes[variable_type]=axis
        # exit
        return axis

    def _redraw_all_series(self):
        # abscissa values — shared across all ordinate steps, will be paired with y below
        abscissa_values=self._abscissa.values[self._zoom_window[0]:self._zoom_window[2]]
        # x0 and x1
        abscissa_min=float(abscissa_values[0])
        abscissa_max=float(abscissa_values[-1])
        try:
            # check decimation algorithm is NONE, if so we can skip decimation and just update the series with the new zoom window data (no need to create new series or do a full redraw)
            if _DECIMATION_ALGORITHM == DecimationAlgorithm.NONE:
                # loop existing series
                for _, (_, ordinate_series) in self._series.items():
                    # recalculated min and max values for this variable across all steps based on the new zoom window
                    new_ordinate_series=[]
                    # loop series data (actual data visible in chart)
                    for ordinate_variant, y_axis, series_list, _, _ in ordinate_series:
                        # min and max value recalculation for the new zoom window
                        min_value=float("inf")
                        max_value=float("-inf")
                        # loop steps
                        for step in range(self._abscissa.steps):
                            # ordinate variant values for this step
                            ordinate_values=ordinate_variant.step_values(step)[self._zoom_window[0]:self._zoom_window[2]]
                            # update series with non-decimated data
                            series_list[step].replaceNp(abscissa_values, ordinate_values)
                            # update min and max values
                            min_value=min(min_value, float(np.min(ordinate_values)))
                            max_value=max(max_value, float(np.max(ordinate_values)))
                        # append to list for later update of the series data and axis ranges after the loop
                        new_ordinate_series.append((ordinate_variant, y_axis, series_list, min_value, max_value))
                    # replace the list contents in-place
                    ordinate_series[:]=new_ordinate_series
                # exit
                return
            # loop existing series
            for _, (_, ordinate_series) in self._series.items():
                # recalculated min and max values for this variable across all steps based on the new zoom window
                new_ordinate_series=[]
                # loop series data (actual data visible in chart)
                for ordinate_variant, y_axis, series_list, _, _ in ordinate_series:
                    # min and max value recalculation for the new zoom window
                    min_value=float("inf")
                    max_value=float("-inf")
                    # loop steps
                    for step in range(self._abscissa.steps):
                        # ordinate variant values for this step
                        ordinate_values=ordinate_variant.step_values(step)[self._zoom_window[0]:self._zoom_window[2]]
                        # decimate x and y jointly so every plotted (x, y) pair maps to the same original sample
                        x_np, y_np=decimate_xy(abscissa_values, ordinate_values, self._decimate_target, _DECIMATION_ALGORITHM)
                        # update series with decimated data
                        series_list[step].replaceNp(x_np, y_np)
                        # update min and max values
                        min_value=min(min_value, float(np.min(y_np)))
                        max_value=max(max_value, float(np.max(y_np)))
                    # append to list for later update of the series data and axis ranges after the loop
                    new_ordinate_series.append((ordinate_variant, y_axis, series_list, min_value, max_value))
                # replace the list contents in-place
                ordinate_series[:]=new_ordinate_series
        finally:
            # resize abscissa axis
            self._component.resizeAbscissa(abscissa_min, abscissa_max)
