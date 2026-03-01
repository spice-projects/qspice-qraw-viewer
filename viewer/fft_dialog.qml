pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls

Item {
    id: root
    anchors.fill: parent

    // context properties injected by Python:
    //   variableNames        : list of variable name strings
    //   windowFunctions      : list of window function name strings
    //   outputTypes          : list of output type name strings
    //   zeroPaddingOptions   : list of zero-padding option strings
    //   freqRangePreview     : string e.g. "0 Hz – 500 kHz (Nyquist)"
    //   binWidthPreview      : string e.g. "9.77 Hz / bin"
    //   abscissaMin          : float — earliest time in seconds
    //   abscissaMax          : float — latest time in seconds
    //   zoomFromTime         : float — left edge of current zoom window
    //   zoomToTime           : float — right edge of current zoom window

    signal dialogAccepted(string variableName, string windowFn, string zeroPad,
                          string output, bool normalize, string rangeMode,
                          real customFrom, real customTo)
    signal dialogRejected()

    // selected range mode: "all" | "zoom" | "custom"
    property string rangeMode: "all"

    Rectangle {
        anchors.fill: parent
        color: "#1a1b1e"
    }

    // -----------------------------------------------------------------------
    // Scrollable content area
    // -----------------------------------------------------------------------
    Flickable {
        id: scrollArea
        anchors { top: parent.top; left: parent.left; right: parent.right; bottom: buttonBar.top }
        contentHeight: formColumn.implicitHeight + 24
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        Column {
            id: formColumn
            anchors { top: parent.top; left: parent.left; right: parent.right; topMargin: 16; leftMargin: 16; rightMargin: 16 }
            spacing: 14

            // ---------------------------------------------------------------
            // Section: Variable
            // ---------------------------------------------------------------
            Text {
                text: "Variable"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Rectangle {
                width: parent.width
                height: 32
                radius: 4
                color: "#23252e"
                border.color: "#3a3d4a"
                border.width: 1

                ComboBox {
                    id: variableCombo
                    anchors { fill: parent; margins: 2 }
                    model: variableNames
                    background: Rectangle { color: "transparent" }
                    contentItem: Text {
                        leftPadding: 8
                        text: variableCombo.displayText
                        color: "#dce8f8"
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    delegate: ItemDelegate {
                        required property string modelData
                        required property int index
                        width: variableCombo.width
                        contentItem: Text {
                            text: modelData
                            color: "#b0b8c8"
                            font.pixelSize: 12
                        }
                        background: Rectangle {
                            color: variableCombo.highlightedIndex === index ? "#3a3d4a" : "#252730"
                        }
                    }
                    popup.background: Rectangle {
                        color: "#252730"
                        border.color: "#3a3d4a"
                        border.width: 1
                        radius: 4
                    }
                }
            }

            // ---------------------------------------------------------------
            // Section: Data Range
            // ---------------------------------------------------------------
            Text {
                text: "Data Range"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Column {
                width: parent.width
                spacing: 6

                // "All points" option
                Rectangle {
                    width: parent.width
                    height: 32
                    radius: 4
                    color: root.rangeMode === "all" ? "#2a4a7a" : "#23252e"
                    border.color: root.rangeMode === "all" ? "#5b9bd5" : "#3a3d4a"
                    border.width: 1

                    Text {
                        anchors { verticalCenter: parent.verticalCenter; left: parent.left; leftMargin: 10 }
                        text: "All abscissa points"
                        color: root.rangeMode === "all" ? "#dce8f8" : "#b0b8c8"
                        font.pixelSize: 12
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: root.rangeMode = "all"
                    }
                }

                // "Current zoom" option
                Rectangle {
                    width: parent.width
                    height: 32
                    radius: 4
                    color: root.rangeMode === "zoom" ? "#2a4a7a" : "#23252e"
                    border.color: root.rangeMode === "zoom" ? "#5b9bd5" : "#3a3d4a"
                    border.width: 1

                    Text {
                        anchors { verticalCenter: parent.verticalCenter; left: parent.left; leftMargin: 10 }
                        text: "Current zoom / visible range"
                        color: root.rangeMode === "zoom" ? "#dce8f8" : "#b0b8c8"
                        font.pixelSize: 12
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: root.rangeMode = "zoom"
                    }
                }

                // "Custom" option
                Rectangle {
                    width: parent.width
                    height: 32
                    radius: 4
                    color: root.rangeMode === "custom" ? "#2a4a7a" : "#23252e"
                    border.color: root.rangeMode === "custom" ? "#5b9bd5" : "#3a3d4a"
                    border.width: 1

                    Text {
                        anchors { verticalCenter: parent.verticalCenter; left: parent.left; leftMargin: 10 }
                        text: "Custom time range"
                        color: root.rangeMode === "custom" ? "#dce8f8" : "#b0b8c8"
                        font.pixelSize: 12
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: root.rangeMode = "custom"
                    }
                }

                // Custom from/to fields — visible only when "custom" is selected
                Row {
                    visible: root.rangeMode === "custom"
                    width: parent.width
                    spacing: 8

                    Column {
                        width: (parent.width - 8) / 2
                        spacing: 4
                        Text { text: "From (s)"; color: "#7a8599"; font.pixelSize: 11 }
                        Rectangle {
                            width: parent.width; height: 30
                            radius: 4
                            color: "#23252e"
                            border.color: fromField.activeFocus ? "#5b9bd5" : "#3a3d4a"
                            border.width: 1
                            TextInput {
                                id: fromField
                                anchors { fill: parent; leftMargin: 8; rightMargin: 4 }
                                text: zoomFromTime.toFixed(9)
                                color: "#dce8f8"
                                font.pixelSize: 12
                                verticalAlignment: Text.AlignVCenter
                                validator: DoubleValidator { bottom: abscissaMin; top: abscissaMax }
                                selectByMouse: true
                            }
                        }
                    }

                    Column {
                        width: (parent.width - 8) / 2
                        spacing: 4
                        Text { text: "To (s)"; color: "#7a8599"; font.pixelSize: 11 }
                        Rectangle {
                            width: parent.width; height: 30
                            radius: 4
                            color: "#23252e"
                            border.color: toField.activeFocus ? "#5b9bd5" : "#3a3d4a"
                            border.width: 1
                            TextInput {
                                id: toField
                                anchors { fill: parent; leftMargin: 8; rightMargin: 4 }
                                text: zoomToTime.toFixed(9)
                                color: "#dce8f8"
                                font.pixelSize: 12
                                verticalAlignment: Text.AlignVCenter
                                validator: DoubleValidator { bottom: abscissaMin; top: abscissaMax }
                                selectByMouse: true
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // Section: Window Function
            // ---------------------------------------------------------------
            Text {
                text: "Window Function"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Rectangle {
                width: parent.width
                height: 32
                radius: 4
                color: "#23252e"
                border.color: "#3a3d4a"
                border.width: 1

                ComboBox {
                    id: windowCombo
                    anchors { fill: parent; margins: 2 }
                    model: windowFunctions
                    background: Rectangle { color: "transparent" }
                    contentItem: Text {
                        leftPadding: 8
                        text: windowCombo.displayText
                        color: "#dce8f8"
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    delegate: ItemDelegate {
                        required property string modelData
                        required property int index
                        width: windowCombo.width
                        contentItem: Text {
                            text: modelData
                            color: "#b0b8c8"
                            font.pixelSize: 12
                        }
                        background: Rectangle {
                            color: windowCombo.highlightedIndex === index ? "#3a3d4a" : "#252730"
                        }
                    }
                    popup.background: Rectangle {
                        color: "#252730"
                        border.color: "#3a3d4a"
                        border.width: 1
                        radius: 4
                    }
                }
            }

            // ---------------------------------------------------------------
            // Section: Zero Padding
            // ---------------------------------------------------------------
            Text {
                text: "Zero Padding"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Rectangle {
                width: parent.width
                height: 32
                radius: 4
                color: "#23252e"
                border.color: "#3a3d4a"
                border.width: 1

                ComboBox {
                    id: zeroPadCombo
                    anchors { fill: parent; margins: 2 }
                    model: zeroPaddingOptions
                    background: Rectangle { color: "transparent" }
                    contentItem: Text {
                        leftPadding: 8
                        text: zeroPadCombo.displayText
                        color: "#dce8f8"
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    delegate: ItemDelegate {
                        required property string modelData
                        required property int index
                        width: zeroPadCombo.width
                        contentItem: Text {
                            text: modelData
                            color: "#b0b8c8"
                            font.pixelSize: 12
                        }
                        background: Rectangle {
                            color: zeroPadCombo.highlightedIndex === index ? "#3a3d4a" : "#252730"
                        }
                    }
                    popup.background: Rectangle {
                        color: "#252730"
                        border.color: "#3a3d4a"
                        border.width: 1
                        radius: 4
                    }
                }
            }

            // ---------------------------------------------------------------
            // Section: Output Type
            // ---------------------------------------------------------------
            Text {
                text: "Output"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Rectangle {
                width: parent.width
                height: 32
                radius: 4
                color: "#23252e"
                border.color: "#3a3d4a"
                border.width: 1

                ComboBox {
                    id: outputCombo
                    anchors { fill: parent; margins: 2 }
                    model: outputTypes
                    background: Rectangle { color: "transparent" }
                    contentItem: Text {
                        leftPadding: 8
                        text: outputCombo.displayText
                        color: "#dce8f8"
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    delegate: ItemDelegate {
                        required property string modelData
                        required property int index
                        width: outputCombo.width
                        contentItem: Text {
                            text: modelData
                            color: "#b0b8c8"
                            font.pixelSize: 12
                        }
                        background: Rectangle {
                            color: outputCombo.highlightedIndex === index ? "#3a3d4a" : "#252730"
                        }
                    }
                    popup.background: Rectangle {
                        color: "#252730"
                        border.color: "#3a3d4a"
                        border.width: 1
                        radius: 4
                    }
                }
            }

            // ---------------------------------------------------------------
            // Section: Options
            // ---------------------------------------------------------------
            Text {
                text: "Options"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            // Normalize checkbox
            Rectangle {
                width: parent.width
                height: 32
                radius: 4
                color: normalizeCheck.checked ? "#2a3a2a" : "#23252e"
                border.color: normalizeCheck.checked ? "#4a9a4a" : "#3a3d4a"
                border.width: 1

                Row {
                    anchors { fill: parent; leftMargin: 10 }
                    spacing: 8

                    CheckBox {
                        id: normalizeCheck
                        anchors.verticalCenter: parent.verticalCenter
                        checked: false
                        indicator: Rectangle {
                            width: 16; height: 16
                            radius: 3
                            anchors.verticalCenter: parent.verticalCenter
                            color: normalizeCheck.checked ? "#4a9a4a" : "#23252e"
                            border.color: normalizeCheck.checked ? "#4a9a4a" : "#5a6070"
                            border.width: 1
                            Text {
                                anchors.centerIn: parent
                                text: "✓"
                                color: "#ffffff"
                                font.pixelSize: 11
                                visible: normalizeCheck.checked
                            }
                        }
                    }

                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Normalize output (peak = 1)"
                        color: "#b0b8c8"
                        font.pixelSize: 12
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: normalizeCheck.checked = !normalizeCheck.checked
                }
            }

            // ---------------------------------------------------------------
            // Section: Preview
            // ---------------------------------------------------------------
            Text {
                text: "Output Preview"
                color: "#7a8599"
                font.pixelSize: 11
                font.capitalization: Font.AllUppercase
                leftPadding: 2
            }

            Rectangle {
                width: parent.width
                implicitHeight: previewColumn.implicitHeight + 16
                radius: 4
                color: "#14151a"
                border.color: "#2a2d35"
                border.width: 1

                Column {
                    id: previewColumn
                    anchors { fill: parent; margins: 8 }
                    spacing: 4

                    Row {
                        spacing: 6
                        Text { text: "Frequency range:"; color: "#7a8599"; font.pixelSize: 11; width: 110 }
                        Text { text: freqRangePreview; color: "#b0b8c8"; font.pixelSize: 11 }
                    }

                    Row {
                        spacing: 6
                        Text { text: "Bin resolution:"; color: "#7a8599"; font.pixelSize: 11; width: 110 }
                        Text { text: binWidthPreview; color: "#b0b8c8"; font.pixelSize: 11 }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Button bar
    // -----------------------------------------------------------------------
    Rectangle {
        id: buttonBar
        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
        height: 44
        color: "#16171e"

        Rectangle {
            anchors { top: parent.top; left: parent.left; right: parent.right }
            height: 1
            color: "#3a3d4a"
        }

        Row {
            anchors { right: parent.right; verticalCenter: parent.verticalCenter; rightMargin: 12 }
            spacing: 8

            Rectangle {
                id: cancelBtn
                width: 80; height: 28
                radius: 4
                color: cancelMouse.containsMouse ? "#2e3040" : "#23252e"
                border.color: "#3a3d4a"
                border.width: 1
                Text {
                    anchors.centerIn: parent
                    text: "Cancel"
                    color: "#b0b8c8"
                    font.pixelSize: 12
                }
                MouseArea {
                    id: cancelMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: root.dialogRejected()
                }
            }

            Rectangle {
                id: runBtn
                width: 80; height: 28
                radius: 4
                color: runMouse.containsMouse ? "#3a6aaa" : "#2a5090"
                border.color: "#5b9bd5"
                border.width: 1
                Text {
                    anchors.centerIn: parent
                    text: "Run"
                    color: "#ffffff"
                    font.pixelSize: 12
                    font.bold: true
                }
                MouseArea {
                    id: runMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        var varName = variableCombo.currentText
                        var winFn = windowCombo.currentText
                        var zp = zeroPadCombo.currentText
                        var out = outputCombo.currentText
                        var norm = normalizeCheck.checked
                        var from = parseFloat(fromField.text)
                        var to = parseFloat(toField.text)
                        root.dialogAccepted(varName, winFn, zp, out, norm, root.rangeMode, from, to)
                    }
                }
            }
        }
    }
}
