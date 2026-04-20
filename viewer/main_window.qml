pragma ComponentBehavior: Bound

import QtQuick
import QtGraphs

Item {
    id: root
    anchors.fill: parent

    property int _activeChartIndex: -1
    property int _activeChartSeriesCount: 0
    property bool fftEnabled: false
    property bool stepToolEnabled: false

    signal horizontalZoom(int chartIndex, real xLeftRatio, real xRightRatio, real zoomFactor)
    signal verticalZoom(int chartIndex, real yTopRatio, real yBottomRatio)
    signal menuZoomToFit(int chartIndex)
    signal menuAutorange(int chartIndex)
    signal menuZoomAbscissaExtent(int chartIndex)
    signal menuAddRemovePlots(int chartIndex)
    signal menuDeleteAllPlots(int chartIndex)
    signal menuAddChart(int chartIndex)
    signal menuDeleteChart(int chartIndex)
    signal menuNewWindow()
    signal menuFft(int chartIndex)
    signal menuStepTool(int chartIndex)
    signal pointerMoved(int chartIndex, real xRatio)
    signal pointerExited(int chartIndex)

    component ChartPanel: Item {
        id: panel

        // index of this panel in the chartsModel - set by the Repeater delegate
        required property int chartIndex

        property int seriesCount: legendModel.count
        property bool legendVisible: false
        readonly property real plotAreaWidth: graphsView.plotArea.width

        signal horizontalZoom(real xLeftRatio, real xRightRatio, real zoomFactor)
        signal verticalZoom(real yTopRatio, real yBottomRatio)
        signal menuZoomToFit
        signal menuAutorange
        signal menuZoomAbscissaExtent
        signal menuAddRemovePlots
        signal menuDeleteAllPlots
        signal menuDeleteChart
        // carries panel-local mouse coords so the root can position the shared menu
        signal menuOpenRequested(real localX, real localY, int seriesCount)
        signal pointerMoved(real xRatio)
        signal pointerExited

        // thin divider drawn above every panel except the first
        Rectangle {
            visible: panel.chartIndex > 0
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
            }
            height: 2
            color: "#12131a"
        }

        Timer {
            id: legendRevealTimer
            interval: 150
            repeat: false
            onTriggered: panel.legendVisible = true
        }

        GraphsView {
            id: graphsView
            marginLeft: 30
            marginRight: 30
            marginBottom: 0
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
                bottom: panelLegend.top
            }

            property string xScale: "lin"
            property string xUnit: ""

            theme: GraphsTheme {
                colorScheme: GraphsTheme.ColorScheme.Dark
                theme: GraphsTheme.Theme.UserDefined
                backgroundColor: "#1a1b1e"
                plotAreaBackgroundColor: "#0d0e10"
                colorStyle: GraphsTheme.ColorStyle.Uniform
                labelTextColor: "#b0b8c8"
                labelFont.pixelSize: 10
                labelBackgroundVisible: false
                labelBorderVisible: false
                gridVisible: true
                grid.mainColor: "#2a2d35"
                grid.subColor: "#1e2028"
                grid.mainWidth: 1
                grid.subWidth: 1
                borderWidth: 0
            }

            axisX: ValueAxis {
                id: axisX
                min: 0
                max: 1
                lineVisible: true
                labelsVisible: true
                titleVisible: false
                titleText: ""
                alignment: Qt.AlignBottom

                labelDelegate: Item {

                    property string text: ""

                    Text {
                        anchors.fill: parent
                        color: "#b0b8c8"
                        font.pixelSize: 10
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignTop
                        text: {
                            // format the X axis label based on the selected scale
                            switch (graphsView.xScale) {
                            case "dec":
                                return graphsView.decadeValueFormatter(graphsView.xUnit, parent.text);
                            case "oct":
                                return graphsView.octaveValueFormatter(graphsView.xUnit, parent.text);
                            default:
                                return graphsView.linearValueFormatter(graphsView.xUnit, parent.text);
                            }
                        }
                    }
                }
            }

            function applyUnit(unit, text, value) {
                // absolute value for prefix selection
                const absValue = Math.abs(value);
                // giga
                if (absValue >= 1e9)
                    return (value / 1e9).toFixed(1) + "G" + unit;
                // mega
                if (absValue >= 1e6)
                    return (value / 1e6).toFixed(1) + "M" + unit;
                // kilo
                if (absValue >= 1e3)
                    return (value / 1e3).toFixed(1) + "k" + unit;
                // base unit
                if (absValue >= 1.0)
                    return value.toFixed(1) + unit;
                // zero
                if (absValue < 1e-15)
                    return "0" + unit;
                // femto
                if (absValue < 1e-12)
                    return (value * 1e15).toFixed(1) + "f" + unit;
                // pico
                if (absValue < 1e-9)
                    return (value * 1e12).toFixed(1) + "p" + unit;
                // nano
                if (absValue < 1e-6)
                    return (value * 1e9).toFixed(1) + "n" + unit;
                // micro
                if (absValue < 1e-3)
                    return (value * 1e6).toFixed(1) + "µ" + unit;
                // milli
                return (value * 1e3).toFixed(1) + "m" + unit;
            }

            function linearValueFormatter(unit, text) {
                // parse value
                var value = parseFloat(text);
                if (isNaN(value))
                    return text;
                // unit
                return applyUnit(unit, text, value);
            }

            function decadeValueFormatter(unit, text) {
                // parse value
                var value = parseFloat(text);
                if (isNaN(value))
                    return text;
                // calculate actual value from decade exponent
                var actual = Math.pow(10, value);
                // unit
                return applyUnit(unit, text, actual);
            }

            function octaveValueFormatter(unit, text) {
                // parse value
                var value = parseFloat(text);
                if (isNaN(value))
                    return text;
                // calculate actual value from octave exponent
                var actual = Math.pow(2, value);
                // unit
                return applyUnit(unit, text, actual);
            }
        }

        Item {
            id: selectionOverlay
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
                bottom: panelLegend.top
            }

            // last mouse X recorded during a pan drag — updated each frame so each delta is incremental
            property real panLastX: 0
            // last mouse Y recorded during a pan drag — updated each frame so each delta is incremental
            property real panLastY: 0

            // map a pixel X within the overlay to a 0-1 plot-area fraction (0=left, 1=right)
            function pixelToXRatio(px) {
                // rectangle
                var r = graphsView.plotArea;
                // compute ratio of pixel X within the plot area, clamped to [0, 1]
                var ratio = (px - r.x) / r.width;
                // clamp and return
                return Math.max(0, Math.min(1, ratio));
            }

            // perform a horizontal zoom around a normalized centre point
            // `center` must be in [0,1] and represents the x-position of the
            // mouse cursor (or any other pivot) expressed as a fraction of the
            // current visible range.  `factor` is the scale factor applied to
            // the window width (<1 zooms in, >1 zooms out).  This routine
            // computes the new [left,right] ratios such that the value at
            // `center` remains fixed on the screen, mimicking the behaviour of
            // most professional plotting applications.
            function applyXZoom(center, factor) {
                // compute raw ratios
                var xr1 = center - center * factor;
                var xr2 = center + (1.0 - center) * factor;
                // enforce non-negative left bound only; right may exceed 1 so Python
                // can distinguish zoom-out from pan
                if (xr1 < 0) {
                    xr1 = 0;
                }

                panel.horizontalZoom(xr1, xr2, factor);
            }

            // left-button drag — pans the X axis left/right
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.LeftButton
                // enable hover so onPositionChanged fires without a button held down
                hoverEnabled: true
                // show grab cursor while hovering so the interaction is discoverable
                cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor

                onPressed: mouse => {
                    // record the starting X and Y so the first positionChanged has a valid reference
                    selectionOverlay.panLastX = mouse.x;
                    selectionOverlay.panLastY = mouse.y;
                }

                onPositionChanged: mouse => {
                    if (pressed) {
                        // compute how far the mouse moved as a fraction of the plot area dimensions
                        var dx = (mouse.x - selectionOverlay.panLastX) / graphsView.plotArea.width;
                        var dy = (mouse.y - selectionOverlay.panLastY) / graphsView.plotArea.height;
                        // update references for the next incremental step
                        selectionOverlay.panLastX = mouse.x;
                        selectionOverlay.panLastY = mouse.y;
                        // dragging right means pulling the data right — shift the horizontal window left
                        panel.horizontalZoom(0 - dx, 1 - dx, 1);
                        // dragging down in screen space means pulling the data down — shift the vertical window up
                        // screen Y is inverted vs data Y so the sign is opposite to horizontal
                        panel.verticalZoom(dy, 1 + dy);
                    }
                    // always report pointer position to the status bar
                    panel.pointerMoved(selectionOverlay.pixelToXRatio(mouse.x));
                }

                onExited: panel.pointerExited()

                onDoubleClicked: panel.menuAutorange()
            }

            // right-button click — asks root to open the shared context menu at this location
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.RightButton
                onClicked: mouse => panel.menuOpenRequested(mouse.x, mouse.y, panel.seriesCount)
            }

            // mouse-wheel — X zoom toward cursor (plain), Y zoom toward cursor (Alt held)
            WheelHandler {
                // include TouchPad so macOS trackpads are handled as well
                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                onWheel: function (event) {
                    // zoom factor: <1 = zoom in, >1 = zoom out; 0.15 per standard detent
                    // log raw wheel movement

                    // compute factor: positive delta will now produce f>1 (zoom out)
                    var f = 1.0 + (event.angleDelta.y / 120) * 0.15;
                    // clamp to avoid inverting the range or zooming out infinitely
                    f = Math.max(0.05, Math.min(4.0, f));
                    if (event.modifiers & Qt.AltModifier) {
                        // vertical zoom centered on cursor Y position
                        var r = graphsView.plotArea;
                        // cursor as 0-1 fraction of plot height in screen space (0=top, 1=bottom)
                        var cy_screen = Math.max(0, Math.min(1, (event.y - r.y) / r.height));
                        // invert to data space so 0=bottom of range, 1=top of range
                        var cy = 1.0 - cy_screen;
                        var yr1 = cy - cy * f;
                        var yr2 = cy + (1.0 - cy) * f;
                        // emit dedicated vertical signal so it never collides with horizontal ratios
                        panel.verticalZoom(yr1, yr2);
                    } else {
                        // horizontal zoom centered on cursor X position
                        var cx = selectionOverlay.pixelToXRatio(event.x);
                        // call helper on the overlay object so it’s in scope
                        selectionOverlay.applyXZoom(cx, f);
                    }
                }
            }
        }

        ListModel {
            id: legendModel
        }

        Rectangle {
            id: panelLegend
            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
            }
            color: "#1a1b1e"
            height: 20
            opacity: panel.legendVisible ? 1 : 0

            Row {
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 16
                Repeater {
                    model: legendModel
                    delegate: Row {
                        required property string seriesName
                        required property string seriesColor
                        spacing: 6
                        Rectangle {
                            width: 4
                            height: 4
                            color: parent.seriesColor
                            anchors.verticalCenter: parent.verticalCenter
                        }
                        Text {
                            text: parent.seriesName
                            color: "#b0b8c8"
                            font.pixelSize: 10
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                }
            }
        }

        Component {
            id: yAxisComponent

            ValueAxis {
                id: valueAxis

                property string yUnit: ""

                lineVisible: true
                labelsVisible: true
                titleVisible: false
                alignment: Qt.AlignLeft
                labelDelegate: Item {
                    property string text: ""
                    Text {
                        anchors.fill: parent
                        color: "#b0b8c8"
                        font.pixelSize: 10
                        horizontalAlignment: valueAxis.alignment === Qt.AlignLeft ? Text.AlignRight : Text.AlignLeft
                        verticalAlignment: Text.AlignVCenter
                        text: graphsView.linearValueFormatter(valueAxis.yUnit, parent.text)
                    }
                }
            }
        }

        function initialize(axisXTitle, axisXUnit, axisXScale, axisXMinValue, axisXMaxValue) {
            // set X title
            axisX.titleText = axisXTitle;
            // unit and scale
            graphsView.xUnit = axisXUnit;
            graphsView.xScale = axisXScale;
            // one tick per decade/octave in log space; ten equal ticks for linear
            axisX.tickInterval = (axisXScale === "lin") ? (axisXMaxValue - axisXMinValue) / 10 : 1.0;
            axisX.subTickCount = (axisXScale === "lin") ? 10 : 0;
            // set X range
            axisX.min = axisXMinValue;
            axisX.max = axisXMaxValue;
        }

        function createYAxis(alignment, unit) {
            // create and return a new ValueAxis instance from the component
            const axis = yAxisComponent.createObject(graphsView);
            // set property values
            axis.alignment = alignment;
            axis.yUnit = unit;
            // use axis
            return axis;
        }

        function resizeAbscissa(axisXMinValue, axisXMaxValue) {
            // update X axis range after a zoom operation
            axisX.min = axisXMinValue;
            axisX.max = axisXMaxValue;
            // one tick per decade/octave in log space; ten equal ticks for linear
            axisX.tickInterval = (graphsView.xScale === "lin") ? (axisXMaxValue - axisXMinValue) / 10 : 1.0;
        }

        function updateGraphsView(seriesToAdd, seriesToRemove) {
            // skip when there are no updates
            if (seriesToAdd.length === 0 && seriesToRemove.length === 0)
                return;
            // avoid repeated repaints while many series are updated in one event
            const wasVisible = graphsView.visible;
            if (wasVisible)
                graphsView.visible = false;
            // collect removals first so transient add/remove overlap does not churn scenegraph state
            const removeLegendNames = {};
            for (var i = 0; i < seriesToRemove.length; i++) {
                // current remove payload
                const removeCurrent = seriesToRemove[i];
                // extract name and series list
                const removeName = removeCurrent[0];
                const removeData = removeCurrent[1];
                // remove every series in this payload
                for (var j = removeData.length - 1; j >= 0; j--)
                    graphsView.removeSeries(removeData[j]);
                // mark legend entries to remove in a single legend scan later
                if (removeName != null)
                    removeLegendNames[removeName] = true;
            }
            // remove legend entries in one reverse pass
            for (var i = legendModel.count - 1; i >= 0; i--) {
                // current legend label
                const legendName = legendModel.get(i)["seriesName"];
                // remove when this label was marked for deletion
                if (removeLegendNames[legendName] === true)
                    legendModel.remove(i);
            }
            // collect additions after removals to minimize intermediate graph states
            const legendEntriesToAdd = [];
            for (var i = 0; i < seriesToAdd.length; i++) {
                // current add payload
                const addCurrent = seriesToAdd[i];
                // extract name, color and series list
                const addName = addCurrent[0];
                const addColor = addCurrent[1];
                const addData = addCurrent[2];
                // add every series for this payload
                for (var j = 0; j < addData.length; j++) {
                    // current series
                    const addSeries = addData[j];
                    // set line color from the palette
                    addSeries.color = addColor;
                    // append to chart
                    graphsView.addSeries(addSeries);
                }
                // queue legend addition with same color; step additions continue using existing legend entries
                if (addName != null)
                    legendEntriesToAdd.push({ seriesName: addName, seriesColor: addColor });
            }
            // append legend entries in one pass
            for (var i = 0; i < legendEntriesToAdd.length; i++)
                legendModel.append(legendEntriesToAdd[i]);
            // restore visibility after batch updates
            if (wasVisible)
                graphsView.visible = true;
            // reveal the legend the first time we plot series
            if (!panel.legendVisible) {
                // reveal the legend after a short delay so the chart has time to paint first
                legendRevealTimer.restart();
            }
        }

        function removeAllSeries() {
            // loop all series in the chart and remove them
            for (var i = graphsView.seriesList.length - 1; i >= 0; i--)
                graphsView.removeSeries(i);
            // clear the legend
            legendModel.clear();
            // reset default Y axis
            graphsView.axisY = null;
        }
    }

    ListModel {
        id: chartsModel
    }

    Column {
        id: chartsColumn
        anchors.fill: parent
        spacing: 2

        Repeater {
            id: chartsRepeater
            model: chartsModel

            delegate: ChartPanel {
                // index is a required property under pragma ComponentBehavior: Bound
                required property int index

                chartIndex: index
                width: chartsColumn.width
                // distribute height equally, accounting for inter-panel spacing
                height: (chartsColumn.height - chartsColumn.spacing * Math.max(0, chartsModel.count - 1)) / Math.max(1, chartsModel.count)

                onHorizontalZoom: (xr1, xr2, f) => root.horizontalZoom(index, xr1, xr2, f)
                onVerticalZoom: (yr1, yr2) => root.verticalZoom(index, yr1, yr2)
                // bubble menu action signals up to root, adding chartIndex
                onMenuZoomToFit: root.menuZoomToFit(index)
                onMenuAutorange: root.menuAutorange(index)
                onMenuZoomAbscissaExtent: root.menuZoomAbscissaExtent(index)
                onMenuAddRemovePlots: root.menuAddRemovePlots(index)
                onMenuDeleteAllPlots: root.menuDeleteAllPlots(index)
                onMenuDeleteChart: root.menuDeleteChart(index)
                // bubble pointer hover signals up to root, adding chartIndex
                onPointerMoved: xRatio => root.pointerMoved(index, xRatio)
                onPointerExited: root.pointerExited(index)
                // position and reveal the single shared context menu on right-click
                onMenuOpenRequested: (localX, localY, sc) => {
                    // map from panel-local coordinates to root-local coordinates
                    var pt = mapToItem(root, localX, localY);
                    root._activeChartIndex = index;
                    root._activeChartSeriesCount = sc;
                    // clamp so the menu never overflows the root boundary
                    contextMenu.x = Math.min(pt.x, root.width - contextMenu.width - 2);
                    contextMenu.y = Math.min(pt.y, root.height - contextMenu.height - 2);
                    // show menu
                    contextMenu.visible = true;
                }
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        visible: contextMenu.visible
        z: 998
        acceptedButtons: Qt.AllButtons
        onPressed: contextMenu.visible = false
    }

    Rectangle {
        id: contextMenu
        visible: false
        z: 999
        width: 210
        height: menuColumn.implicitHeight + 8
        color: "#252730"
        border.color: "#3a3d4a"
        border.width: 1
        radius: 4

        Column {
            id: menuColumn
            anchors {
                top: parent.top
                left: parent.left
                topMargin: 4
            }

            ContextMenuItem {
                itemText: "Zoom to Fit"
                onTriggered: root.menuZoomToFit(root._activeChartIndex)
            }
            ContextMenuItem {
                itemText: "Autorange"
                onTriggered: root.menuAutorange(root._activeChartIndex)
            }
            ContextMenuItem {
                itemText: "Zoom Abscissa Extent"
                onTriggered: root.menuZoomAbscissaExtent(root._activeChartIndex)
            }
            ContextMenuSeparator {}
            ContextMenuItem {
                itemText: "Add/Remove Plots"
                onTriggered: root.menuAddRemovePlots(root._activeChartIndex)
            }
            ContextMenuItem {
                itemText: "Delete All Plots"
                enabled: root._activeChartSeriesCount > 0
                onTriggered: root.menuDeleteAllPlots(root._activeChartIndex)
            }
            ContextMenuSeparator {}
            ContextMenuItem {
                itemText: "FFT..."
                enabled: root.fftEnabled && root._activeChartSeriesCount > 0
                onTriggered: root.menuFft(root._activeChartIndex)
            }
            ContextMenuItem {
                itemText: "Step Tool..."
                enabled: root.stepToolEnabled
                onTriggered: root.menuStepTool(root._activeChartIndex)
            }
            ContextMenuSeparator {}
            ContextMenuItem {
                itemText: "Add Chart"
                onTriggered: root.menuAddChart(root._activeChartIndex)
            }
            ContextMenuItem {
                itemText: "Delete Chart"
                enabled: chartsModel.count > 1
                onTriggered: root.menuDeleteChart(root._activeChartIndex)
            }
            ContextMenuSeparator {}
            ContextMenuItem {
                itemText: "New Window"
                onTriggered: root.menuNewWindow()
            }
        }
    }

    component ContextMenuItem: Rectangle {
        id: itemRoot
        required property string itemText
        signal triggered
        width: contextMenu.width
        implicitHeight: 26
        color: itemMouse.containsMouse ? "#3a3d4a" : "transparent"
        radius: 2
        Text {
            anchors.fill: parent
            text: itemRoot.itemText
            color: itemRoot.enabled ? "#b0b8c8" : "#4a5060"
            font.pixelSize: 12
            leftPadding: 12
            verticalAlignment: Text.AlignVCenter
        }
        MouseArea {
            id: itemMouse
            anchors.fill: parent
            hoverEnabled: true
            enabled: itemRoot.enabled
            onClicked: {
                contextMenu.visible = false;
                itemRoot.triggered();
            }
        }
    }

    // reusable separator
    component ContextMenuSeparator: Rectangle {
        width: contextMenu.width
        implicitHeight: 9
        color: "transparent"
        Rectangle {
            anchors.centerIn: parent
            width: parent.width - 8
            height: 1
            color: "#3a3d4a"
        }
    }

    function addChart() {
        // append a new chart panel entry to the model; the Repeater reacts immediately
        chartsModel.append({});
    }

    function removeChart(chartIndex) {
        // remove the panel at the given index; remaining panels reflow automatically
        chartsModel.remove(chartIndex);
    }

    function getChart(chartIndex) {
        // return the live ChartPanel item so Python can call initialize / addSeries etc.
        return chartsRepeater.itemAt(chartIndex);
    }
}
