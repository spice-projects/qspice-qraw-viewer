import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Q_ARG, QMetaObject, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickItem, QQuickView, QSGNode, QSGSimpleTextureNode, QSGTexture
from PySide6.QtWidgets import QMainWindow, QWidget

from .add_plot_dialog import AddPlotDialog
from .app_open import open_qraw_as_window
from .color_palette import SERIES_COLOR_PALETTE
from .decimation_algorithm import decimate_xy, DecimationAlgorithm
from .expression import Expression
from .expression_manager import ExpressionManager
from .qraw_file import QRawFile
from .step_tool_dialog import StepToolDialog
from .window import load_app_icon, log_screen_info, unregister_child_window

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "smith_chart_window.qml"

# background color matching the chart dark theme
_BG = "#1a1b1e"

R_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]  # constant-resistance
X_VALUES = [0.2, 0.5, 1.0, 2.0, 5.0]  # constant-reactance (±)
SEGS = 128  # tessellation per circle
GRID_COLOR = QColor(83, 74, 183, 160)  # purple-ish
REACT_COLOR = QColor(15, 110, 86, 160)  # teal


# this is the import name and version that QML uses to identify the module containing the SmithGridItem type; it does not need to match the Python module name or version, but it must match the import statement in the QML file
QML_IMPORT_NAME = "SmithChart"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class SmithGridItem(QQuickItem):

    def __init__(self, parent: QQuickItem | None = None):
        super().__init__(parent)
        # initialize state
        self.setFlag(QQuickItem.Flag.ItemHasContents, True)
        self._texture: QSGTexture | None = None
        self._dirty = True
        self._image_size = (0, 0)

    def geometryChange(self, new: QRectF | QRect, old: QRectF | QRect):
        super().geometryChange(new, old)
        self._dirty = True
        self.update()

    def updatePaintNode(self, old_node: QSGNode, _data: QQuickItem.UpdatePaintNodeData) -> QSGNode:
        # check dirty flag
        if not self._dirty:
            return old_node
        # reset dirty flag
        self._dirty = False
        # get dimensions in pixels
        W, H = int(self.width()), int(self.height())
        if W <= 0 or H <= 0:
            return old_node
        # only re-render grid if size changed; otherwise reuse existing texture and node tree (GPU-based rendering is very fast after the initial rasterization)
        if (W, H) != self._image_size:
            # store image size for future change detection
            self._image_size = (W, H)
            # create image
            image = self._render_grid(W, H)
            # delete old texture if it exists to free GPU memory; this is important to avoid leaks when resizing the window multiple times, as each new texture consumes GPU resources until the old one is released
            if self._texture:
                self._texture.deleteLater()
            # upload new texture to GPU; this is a one-time cost that can be expensive for large images, but it allows for very fast rendering in subsequent frames since the GPU can efficiently draw the pre-rasterized grid without needing to execute CPU-based drawing commands each time
            self._texture = self.window().createTextureFromImage(image)
        # textured rectangle, fully GPU-composited
        node = old_node or QSGSimpleTextureNode()
        node.setTexture(self._texture)
        node.setRect(QRectF(0, 0, W, H))
        node.setFiltering(QSGTexture.Filtering.Linear)
        # exit
        return node

    def _render_grid(self, W: int, H: int) -> QImage:
        # create transparent image and paint the grid onto it using QPainter; this is a CPU-based operation that generates a rasterized representation of the Smith chart grid, which can then be efficiently rendered by the GPU as a texture in subsequent frames
        image = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        # calculate center and radius of the unit disk in pixel coordinates; the grid will be drawn within this disk, and the clipping region will ensure that any parts of the circles that extend beyond the unit disk are not rendered, creating the characteristic shape of the Smith chart
        cx, cy = W / 2, H / 2
        R = min(W, H) / 2 * 0.90
        # set up QPainter for anti-aliased drawing; this allows for smoother and visually appealing rendering of the circles and arcs that make up the Smith chart grid, at the cost of increased CPU time during the rasterization process, which is why we want to minimize how often this needs to be done by caching the resulting texture
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # clip everything to the unit disk
        clip = QPainterPath()
        clip.addEllipse(cx - R, cy - R, R * 2, R * 2)
        painter.setClipPath(clip)
        # resistance circles
        pen_r = QPen(QColor(83, 74, 183, 160), 1.0)
        painter.setPen(pen_r)
        for r in R_VALUES:
            cr = R / (r + 1)
            ccx = cx + r / (r + 1) * R
            painter.drawEllipse(QRectF(ccx - cr, cy - cr, cr * 2, cr * 2))
        # reactance arcs (clipping handles the truncation automatically)
        pen_x = QPen(QColor(15, 110, 86, 160), 1.0)
        painter.setPen(pen_x)
        for x in X_VALUES:
            for sign in [1, -1]:
                xv = sign * x
                cr = R / abs(xv)
                ccx = cx + R  # Γ_re=1 → right edge of unit disk
                ccy = cy - R / xv
                painter.drawEllipse(QRectF(ccx - cr, ccy - cr, cr * 2, cr * 2))
        # outer boundary
        painter.setPen(QPen(QColor(120, 120, 120, 200), 1.5))
        painter.drawEllipse(QRectF(cx - R, cy - R, R * 2, R * 2))
        # real axis
        painter.setPen(QPen(QColor(120, 120, 120, 120), 0.7))
        painter.drawLine(int(cx - R), int(cy), int(cx + R), int(cy))
        # end
        painter.end()
        # exit
        return image


@QmlElement
class SmithTraceItem(QQuickItem):

    def __init__(self, parent: QQuickItem | None = None):
        super().__init__(parent)
        # initialize state
        self.setFlag(QQuickItem.Flag.ItemHasContents, True)
        self._traces = []
        self._texture = None
        self._dirty = True

    def geometryChange(self, new: QRectF | QRect, old: QRectF | QRect):
        super().geometryChange(new, old)
        self._dirty = True
        self.update()

    def updatePaintNode(self, old_node: QSGNode, _data: QQuickItem.UpdatePaintNodeData) -> QSGNode:
        # get dimensions in pixels
        W, H = int(self.width()), int(self.height())
        if W <= 0 or H <= 0:
            return old_node
        # check dirty flag
        if self._dirty:
            # reset dirty flag
            self._dirty = False
            # create transparent image and paint the grid onto it using QPainter; this is a CPU-based operation that generates a rasterized representation of the Smith chart grid, which can then be efficiently rendered by the GPU as a texture in subsequent frames
            image = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(Qt.GlobalColor.transparent)
            # calculate center and radius of the unit disk in pixel coordinates; the grid will be drawn within this disk, and the clipping region will ensure that any parts of the circles that extend beyond the unit disk are not rendered, creating the characteristic shape of the Smith chart
            cx, cy = W / 2, H / 2
            R = min(W, H) / 2 * 0.90
            # set up QPainter for anti-aliased drawing; this allows for smoother and visually appealing rendering of the circles and arcs that make up the Smith chart grid, at the cost of increased CPU time during the rasterization process, which is why we want to minimize how often this needs to be done by caching the resulting texture
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # draw traces
            for pts_gamma, _, color in self._traces:
                # skip if not enough points to draw a trace; this can happen when the trace is first added and the data is still being loaded or processed, allowing us to avoid unnecessary drawing operations and potential errors from trying to render incomplete data
                if pts_gamma is None or len(pts_gamma) < 2:
                    continue
                # convert trace points from complex Γ coordinates to pixel coordinates; this involves mapping the complex plane of the Smith chart (where the real and imaginary parts of Γ correspond to the horizontal and vertical axes, respectively) to the pixel coordinate system of the image, taking into account the center and radius of the unit disk to ensure that the trace is accurately positioned on the chart
                px = self._get_pixel_pts(pts_gamma, W, H, cx, cy, R)
                n = len(px)
                # set up pen for drawing the trace with the specified color and width; using a round cap style helps to create smoother endpoints for the trace, which can
                pen = QPen(color, 2.0)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                # create a QPolygonF from the pixel coordinates of the trace points; this is a convenient way to represent the series of points that make up the trace, and it can be efficiently rendered as a connected series of line segments using QPainter's drawPolyline method, which is ideal for visualizing the continuous nature of the trace on the Smith chart
                poly = QPolygonF([QPointF(float(px[i, 0]), float(px[i, 1])) for i in range(n)])
                # draw the trace as a connected series of line segments defined by the points in the trace, which allows us to visualize the behavior of the impedance or reflection coefficient across the frequency range on the Smith chart; this is a key part of the visualization that helps users understand how their circuit behaves in terms of impedance matching and other RF characteristics
                painter.drawPolyline(poly)
            # end QPainter
            painter.end()
            # delete old texture
            if self._texture:
                self._texture.deleteLater()
            # upload new texture to GPU; this is a one-time cost that can be expensive for large images
            self._texture = self.window().createTextureFromImage(image)
        # reuse existing node and texture since nothing changed; this allows for very fast rendering of the trace on top of the grid after the initial rasterization
        node = old_node or QSGSimpleTextureNode()
        node.setTexture(self._texture)
        node.setRect(QRectF(0, 0, W, H))
        node.setFiltering(QSGTexture.Filtering.Linear)
        # exit
        return node

    def _get_pixel_pts(self, pts_gamma, W, H, cx, cy, R):
        # convert trace points from complex Γ coordinates to pixel coordinates; this involves mapping the complex plane of the Smith chart (where the real and imaginary parts of Γ correspond to the horizontal and vertical axes, respectively) to the pixel coordinate system of the image, taking into account the center and radius of the unit disk to ensure that the trace is accurately positioned on the chart; this transformation is essential for correctly rendering the trace on top of the grid, as it ensures that the points are placed in the correct locations relative to the underlying Smith chart structure
        px = np.empty_like(pts_gamma)
        px[:, 0] = cx + pts_gamma[:, 0] * R
        px[:, 1] = cy - pts_gamma[:, 1] * R
        # decimation using Ramer-Douglas-Peucker algorithm to reduce the number of points while preserving the overall shape of the trace; this is important for performance when rendering traces with a large number of points, as it allows us to significantly reduce the number of line segments that need to be drawn without losing important visual information about the trace's behavior
        keep = decimate_xy(px, epsilon=1.0)
        # exit
        return px[keep]

    @Slot("QVariant")
    def plot(self, series):
        logger.debug("Plotting trace with %d points", len(series))

class SmithChartWindow(QMainWindow):

    def __init__(self, qraw_file: QRawFile):
        super().__init__()
        # load and set the application icon
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        # keep a reference to the source qraw object for creating secondary windows
        self._qraw_file = qraw_file
        # extract information from file
        self._abscissa = qraw_file.abscissa
        self._abscissa_scale = qraw_file.abscissa_scale
        self._expression_manager = qraw_file.expression_manager
        self._step_information = qraw_file.step_information
        self._selected_steps: set[int] = set(range(self._step_information.length))
        self._plot_suggestions = qraw_file.get_plot_suggestions()
        # current visualization state
        self._expressions: dict[Expression, tuple[QColor]] = {}
        # set window title to include the loaded filename
        self.setWindowTitle(f"{qraw_file.chart_type} - {qraw_file.filename.name}")
        # apply dark background stylesheet to the window chrome
        self.setStyleSheet(f"QMainWindow {{ background: {_BG}; }}")
        # single QQuickView hosts the entire multi-chart scene — one Metal swap chain
        self._qml_view = QQuickView()
        self._qml_view.statusChanged.connect(self._on_qml_ready)
        self._qml_view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._qml_view.setColor(QColor(_BG))
        self._qml_view.setSource(QUrl.fromLocalFile(str(_QML_FILE)))
        # embed the single QWindow into the main window's central area
        self._container = QWidget.createWindowContainer(self._qml_view, self)
        self.setCentralWidget(self._container)        
        # next color index for new series
        self._next_color_index = 0

    def sizeHint(self):
        return QSize(1200, 800)

    def closeEvent(self, event) -> None:
        # release application-level child window ownership when this window closes
        unregister_child_window(self)
        # delegate to the Qt base class when available
        close_event = getattr(super(), "closeEvent", None)
        if close_event is not None:
            close_event(event)

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # qml view root object
        self._root = self._qml_view.rootObject()
        # set window-level menu capability flags using built-in bool to avoid passing numpy.bool into QML properties
        self._root.setProperty("stepToolVisible", bool(self._step_information.length > 1))
        # connect signals from QML to Python handlers
        self._root.menuAddRemovePlots.connect(self._on_menu_add_remove_plots)
        self._root.menuDeleteAllPlots.connect(self._on_menu_delete_all_plots)
        self._root.menuStepTool.connect(self._on_menu_step_tool)
        # populate chart after the event loop starts so the window is visible first
        QTimer.singleShot(0, self._populate_charts)
        # log screen information for debugging purposes
        if logger.isEnabledFor(logging.DEBUG):
            QTimer.singleShot(0, lambda: log_screen_info(self.screen()))

    @Slot()
    def _on_menu_add_remove_plots(self):
        # log information
        logger.debug("User requested adding/removing plots on smith chart")
        # open the add plot dialog, only show S11 and S22 expressions
        dialog = AddPlotDialog(self, self._expression_manager, list(self._expressions.keys()), allow_custom_expressions=False, expression_filter=lambda expression: expression.complex)
        # exit if the user cancelled
        if dialog.exec() != AddPlotDialog.DialogCode.Accepted:
            return
        # plot selected expressions on the chart
        self._add_plots(dialog.selected_expressions)

    @Slot()
    def _on_menu_delete_all_plots(self):
        # log information
        logger.debug("User requested deleting all plots on smith chart")
        # clear chart
        self._clear()

    @Slot()
    def _on_menu_step_tool(self):
        # log information
        logger.debug("User requested step tool on smith chart")
        # get selected steps for this chart, make a copy
        selected_steps = set(self._selected_steps)
        # open step tool dialog
        dialog = StepToolDialog(self, self._step_information, selected_steps)
        # exit if the user canceled
        if dialog.exec() != StepToolDialog.DialogCode.Accepted:
            return
        # store selected steps for later filtering phase
        self._selected_steps = dialog.selected_steps
        # auto range axes
        # chart.auto_range()

    def _populate_charts(self):
        # loop suggestions — each suggestion carries its own chart type
        for suggestion in self._plot_suggestions:
            # append plot
            self._add_plots(suggestion.expressions)
            # we are visualizing a single Smith Chart, so exit after the first suggestion (more than one expression is supported)
            break

    def _add_plots(self, expressions: set[Expression]):
        # find the SmithTraceItem instance
        trace_item = self._root.findChild(SmithTraceItem)
        if trace_item:
            # remove any expressions that are no longer selected for plotting
            for expression in list(self._expressions.keys()):
                # check it not in set
                if expression not in expressions:
                    # log information
                    logger.debug("Removing series for expression [%s] from Smith chart, steps: %s", expression.name, list(rendered_series.keys()))
                    # remove from state
                    del self._expressions[expression]
            # loop expressions to plot
            for expression in expressions:
                # check we are already plotting this expression
                if expression in self._expressions:
                    continue
                # assign next color in palette
                color = QColor(SERIES_COLOR_PALETTE[self._next_color_index % len(SERIES_COLOR_PALETTE)])
                # update index
                self._next_color_index += 1

                # loop steps
                for step in self._selected_steps:
                                    
                # append to state
                self._expressions[expression] = (color,)
            # plot expressions
            QMetaObject.invokeMethod(trace_item, "plot", Q_ARG("QVariant", self._expressions))

    def _clear(self):
        ...
