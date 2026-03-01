pragma ComponentBehavior: Bound

import QtQuick
import QtGraphs

Item {
    id: root
    anchors.fill: parent

    // context properties set by Python:
    //   seriesLabel  : string — variable name used in the legend
    //   yAxisLabel   : string — unit for the Y axis ("dB", "°", or "")
    //   freqMin      : real   — minimum frequency (Hz)
    //   freqMax      : real   — maximum frequency (Hz)
    //   yMin         : real   — minimum Y value
    //   yMax         : real   — maximum Y value

    signal exportCsvRequested()
    signal closeRequested()

    function addSeries(series) {
        series.axisY = axisY
        graphsView.addSeries(series)
        // set axis ranges
        axisX.min = freqMin
        axisX.max = freqMax
        var range = yMax - yMin
        axisY.min = yMin - range * 0.05
        axisY.max = yMax + range * 0.05
        axisX.tickInterval = (freqMax - freqMin) / 10
        axisY.tickInterval = range > 0 ? range * 1.1 / 8 : 1
        // show legend
        legendText.visible = true
    }

    Rectangle {
        anchors.fill: parent
        color: "#1a1b1e"
    }

    GraphsView {
        id: graphsView
        anchors { top: parent.top; left: parent.left; right: parent.right; bottom: toolbar.top }
        marginLeft: 30
        marginRight: 30
        marginBottom: 0

        theme: GraphsTheme {
            colorScheme: GraphsTheme.ColorScheme.Dark
            theme: GraphsTheme.Theme.UserDefined
            backgroundColor: "#1a1b1e"
            plotAreaBackgroundColor: "#0d0e10"
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
            titleText: "Frequency (Hz)"

            labelDelegate: Item {
                property string text: ""
                Text {
                    anchors.fill: parent
                    color: "#b0b8c8"
                    font.pixelSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignTop
                    text: {
                        var value = parseFloat(parent.text)
                        if (isNaN(value)) return parent.text
                        var absValue = Math.abs(value)
                        if (absValue >= 1e9)  return (value / 1e9).toPrecision(3)  + " GHz"
                        if (absValue >= 1e6)  return (value / 1e6).toPrecision(3)  + " MHz"
                        if (absValue >= 1e3)  return (value / 1e3).toPrecision(3)  + " kHz"
                        if (absValue < 1e-3)  return (value * 1e6).toPrecision(3)  + " µHz"
                        return parent.text + " Hz"
                    }
                }
            }
        }

        ValueAxis {
            id: axisY
            min: 0
            max: 1
            lineVisible: true
            labelsVisible: true
            titleVisible: false
            titleText: ""
            alignment: Qt.AlignLeft

            property string yUnit: yAxisLabel

            labelDelegate: Item {
                property string text: ""
                Text {
                    anchors.fill: parent
                    color: "#b0b8c8"
                    font.pixelSize: 10
                    horizontalAlignment: Text.AlignRight
                    verticalAlignment: Text.AlignVCenter
                    text: {
                        var value = parseFloat(parent.text)
                        if (isNaN(value)) return parent.text
                        var unit = axisY.yUnit
                        if (unit === "") return parent.text
                        return parent.text + " " + unit
                    }
                }
            }
        }
    }

    // Legend
    Text {
        id: legendText
        visible: false
        anchors { bottom: graphsView.bottom; horizontalCenter: graphsView.horizontalCenter; bottomMargin: 6 }
        text: seriesLabel
        color: "#f77f00"
        font.pixelSize: 11
    }

    // Bottom toolbar
    Rectangle {
        id: toolbar
        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
        height: 40
        color: "#16171e"

        Rectangle {
            anchors { top: parent.top; left: parent.left; right: parent.right }
            height: 1
            color: "#3a3d4a"
        }

        Row {
            anchors { left: parent.left; verticalCenter: parent.verticalCenter; leftMargin: 12 }
            spacing: 8

            // Export CSV button
            Rectangle {
                id: exportBtn
                width: 100; height: 28
                radius: 4
                color: exportMouse.containsMouse ? "#2e3040" : "#23252e"
                border.color: "#3a3d4a"
                border.width: 1
                Text {
                    anchors.centerIn: parent
                    text: "Export CSV"
                    color: "#b0b8c8"
                    font.pixelSize: 12
                }
                MouseArea {
                    id: exportMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: root.exportCsvRequested()
                }
            }
        }

        Row {
            anchors { right: parent.right; verticalCenter: parent.verticalCenter; rightMargin: 12 }

            Rectangle {
                width: 80; height: 28
                radius: 4
                color: closeMouse.containsMouse ? "#3a6aaa" : "#2a5090"
                border.color: "#5b9bd5"
                border.width: 1
                Text {
                    anchors.centerIn: parent
                    text: "Close"
                    color: "#ffffff"
                    font.pixelSize: 12
                    font.bold: true
                }
                MouseArea {
                    id: closeMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: root.closeRequested()
                }
            }
        }
    }
}
