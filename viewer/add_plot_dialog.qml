pragma ComponentBehavior: Bound

import QtQuick

Item {
    id: root
    anchors.fill: parent

    property var expressions: []
    property var selectedIndices: []
    property var selectionState: ({})
    property string filterText: ""

    readonly property var filteredExpressions: {
        var text = filterText.toLowerCase()
        if (text === "") return root.expressions
        return root.expressions.filter(function(e) {
            return String(e[0]).toLowerCase().indexOf(text) !== -1
        })
    }

    signal dialogAccepted()
    signal dialogRejected()
    signal selectionChanged(string expression, bool selected)

    function initialize(expressions) {
        root.expressions = expressions
        var state = {}
        for (var i = 0; i < expressions.length; i++) {
            state[expressions[i][0]] = expressions[i][1]
        }
        root.selectionState = state
    }

    Rectangle {
        anchors.fill: parent
        color: "#1a1b1e"
    }

    Text {
        id: titleLabel
        anchors { top: parent.top; left: parent.left; right: parent.right; topMargin: 12; leftMargin: 14 }
        text: "Select one or more expressions to plot:"
        color: "#b0b8c8"
        font.pixelSize: 13
    }

    // filter input bar
    Rectangle {
        id: filterBar
        anchors { top: titleLabel.bottom; left: parent.left; right: parent.right; topMargin: 8; leftMargin: 10; rightMargin: 10 }
        height: 28
        radius: 4
        color: "#23252e"
        border.color: filterInput.activeFocus ? "#5b9bd5" : "#3a3d4a"
        border.width: 1

        TextInput {
            id: filterInput
            anchors { verticalCenter: parent.verticalCenter; left: parent.left; right: clearBtn.left; leftMargin: 8; rightMargin: 4 }
            color: "#dce8f8"
            font.pixelSize: 12
            clip: true
            onTextChanged: root.filterText = filterInput.text

            // placeholder text shown when empty
            Text {
                anchors.fill: parent
                text: "Filter expressions..."
                color: "#555b6a"
                font.pixelSize: 12
                visible: filterInput.text === ""
            }
        }

        // clear button — only visible while filter text is non-empty
        Rectangle {
            id: clearBtn
            anchors { right: parent.right; verticalCenter: parent.verticalCenter; rightMargin: 6 }
            width: 16; height: 16
            radius: 8
            color: clearMouse.containsMouse ? "#3a3d4a" : "transparent"
            visible: filterInput.text !== ""

            Text {
                anchors.centerIn: parent
                text: "\u00D7"
                color: "#b0b8c8"
                font.pixelSize: 13
            }

            MouseArea {
                id: clearMouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: filterInput.text = ""
            }
        }
    }

    GridView {
        id: grid
        anchors {
            top: filterBar.bottom
            left: parent.left; right: parent.right; bottom: buttonBar.top
            margins: 10
            topMargin: 8
        }
        cellWidth: 180
        cellHeight: 28
        clip: true

        model: root.filteredExpressions

        delegate: Item {

            id: cellItem
            width: grid.cellWidth
            height: grid.cellHeight

            required property int index
            required property var modelData

            property string expression: String(modelData[0])
            // read selection state from the root map so toggling persists across filter changes
            property bool selected: root.selectionState[cellItem.expression] === true

            Rectangle {
                anchors { fill: parent; margins: 3 }
                radius: 4
                color: cellItem.selected ? "#2a4a7a" : "#23252e"
                border.color: cellItem.selected ? "#5b9bd5" : "#3a3d4a"
                border.width: 1

                Text {
                    anchors { verticalCenter: parent.verticalCenter; left: parent.left; leftMargin: 8; right: parent.right; rightMargin: 4 }
                    text: cellItem.expression
                    color: cellItem.selected ? "#dce8f8" : "#b0b8c8"
                    font.pixelSize: 12
                    elide: Text.ElideRight
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onEntered: parent.color = cellItem.selected ? "#2a4a7a" : "#2e3040"
                    onExited:  parent.color = cellItem.selected ? "#2a4a7a" : "#23252e"
                    onClicked: {
                        // compute new selection value by inverting the current state
                        var newSelected = !root.selectionState[cellItem.expression]
                        // replace the entire map object so QML detects the change and re-evaluates bindings
                        var updated = Object.assign({}, root.selectionState)
                        updated[cellItem.expression] = newSelected
                        root.selectionState = updated
                        // emit signal to notify of selection change
                        root.selectionChanged(cellItem.expression, newSelected)
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Button bar — OK and Cancel
    // -------------------------------------------------------------------------
    Rectangle {
        id: buttonBar
        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
        height: 44
        color: "#16171e"

        // thin top divider
        Rectangle {
            anchors { top: parent.top; left: parent.left; right: parent.right }
            height: 1
            color: "#3a3d4a"
        }

        Row {
            anchors { right: parent.right; verticalCenter: parent.verticalCenter; rightMargin: 12 }
            spacing: 8

            // cancel button
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

            // ok button
            Rectangle {
                id: okBtn
                width: 80; height: 28
                radius: 4
                color: okMouse.containsMouse ? "#3a6aaa" : "#2a5090"
                border.color: "#5b9bd5"
                border.width: 1
                Text {
                    anchors.centerIn: parent
                    text: "OK"
                    color: "#ffffff"
                    font.pixelSize: 12
                    font.bold: true
                }
                MouseArea {
                    id: okMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: root.dialogAccepted()
                }
            }
        }
    }
}
