pragma ComponentBehavior: Bound

import QtQuick

Item {
    id: root
    anchors.fill: parent

    property var expressions: []
    property var selectedIndices: []

    signal dialogAccepted()
    signal dialogRejected()
    signal selectionChanged(string expression, bool selected)

    function initialize(expressions) {
        root.expressions = expressions
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

    GridView {
        id: grid
        anchors {
            top: titleLabel.bottom
            left: parent.left; right: parent.right; bottom: buttonBar.top
            margins: 10
            topMargin: 8
        }
        cellWidth: 180
        cellHeight: 28
        clip: true

        model: root.expressions

        delegate: Item {

            id: cellItem
            width: grid.cellWidth
            height: grid.cellHeight

            required property int index
            required property var modelData

            property string expression: String(modelData[0])
            property bool selected: Boolean(modelData[1])

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
                        // toggle selection state
                        cellItem.selected = !cellItem.selected
                        // emit signal to notify of selection change, passing the expression name and new selection state
                        root.selectionChanged(cellItem.expression, cellItem.selected)
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
