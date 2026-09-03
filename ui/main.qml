import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root

    width: 1200
    height: 900
    visible: true
    minimumWidth: 1200
    minimumHeight: 900

    title: "Automotive AI Dashboard"

    onClosing: function(close) {
        vehicleBridge.shutdown()
        close.accepted = true
    }

    GridLayout {
        anchors.fill: parent
        anchors.margins: 10

        rows: 6
        columns: 4

        rowSpacing: 10
        columnSpacing: 10

        GroupBox {

            title: "Assistant Control"
            Layout.columnSpan: parent.columns
            Layout.fillWidth: true

            RowLayout {

                anchors.fill: parent

                Switch {

                    id: eventProcessingSwitch

                    checked: false

                    text: checked
                        ? "Event Processing ENABLED"
                        : "Event Processing DISABLED"

                    onCheckedChanged: {
                        vehicleBridge.eventProcessingChanged(
                            checked
                        )
                    }
                }
                
            }
        }
        GroupBox {
            title: "Driving Behaviour"

            Layout.fillWidth: true
            Layout.fillHeight: true

            Column {
                ButtonGroup {
                    id: drivingBehaviourGroupBox

                    onCheckedButtonChanged: {
                        if (drivingBehaviourGroupBox.checkedButton) {
                            vehicleBridge.drivingBehaviorChanged(drivingBehaviourGroupBox.checkedButton.text)
                        }
                    }
                }
                RadioButton { 
                    text: "Normal" 
                    ButtonGroup.group: drivingBehaviourGroupBox 
                    checked: true
                }
                RadioButton {
                    text: "Safe" 
                    ButtonGroup.group: drivingBehaviourGroupBox
                }
                RadioButton {
                    text: "Aggressive"
                    ButtonGroup.group: drivingBehaviourGroupBox
                }
                RadioButton {
                    text: "Non Reactive"
                    ButtonGroup.group: drivingBehaviourGroupBox
                }
            }
        }

        GroupBox {
            title: "Driver Activity"

            Column {

                ButtonGroup {
                    id: activityGroup

                    onCheckedButtonChanged: {
                        if (activityGroup.checkedButton) {
                            vehicleBridge.driverActivityChanged(activityGroup.checkedButton.text)
                        }
                    }
                }

                RadioButton {
                    text: "Idle"
                    ButtonGroup.group: activityGroup
                    checked: true
                }

                RadioButton {
                    text: "Talking"
                    ButtonGroup.group: activityGroup
                }

                RadioButton {
                    text: "Driving"
                    ButtonGroup.group: activityGroup
                }

                RadioButton {
                    text: "Eating"
                    ButtonGroup.group: activityGroup
                }

                RadioButton {
                    text: "On the phone"
                    ButtonGroup.group: activityGroup
                }

                RadioButton {
                    text: "Sleeping"
                    ButtonGroup.group: activityGroup
                }
            }
        }

        GroupBox {
            id: moodGroupBox
            title: "Driver Mood"

            Layout.fillWidth: true
            Layout.fillHeight: true

            Column {
                ButtonGroup {
                    id: moodGroup

                    onCheckedButtonChanged: {
                        if (moodGroup.checkedButton) {
                            vehicleBridge.driverMoodChanged(moodGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "Neutral"
                    ButtonGroup.group: moodGroup
                    checked: true
                }
                RadioButton {
                    text: "Happy"
                    ButtonGroup.group: moodGroup
                }
                RadioButton {
                    text: "Sad"
                    ButtonGroup.group: moodGroup
                }
                RadioButton {
                    text: "Worried"
                    ButtonGroup.group: moodGroup
                }
                RadioButton {
                    text: "Scared"
                    ButtonGroup.group: moodGroup
                }
            }
        }

        GroupBox {

            title: "Driver Status"

            Layout.fillWidth: true

            ColumnLayout {

                anchors.fill: parent

                Column {

                    RowLayout {
                        Label {
                            text: "Fatigue Score"
                        }

                        Slider {

                            id: fatigueSlider

                            from: 0
                            to: 1
                            value: 0

                            onPressedChanged: {
                                if (!pressed) {
                                    vehicleBridge.fatigueChanged(value)
                                }
                            }

                        }

                        Label {
                            text: fatigueSlider.value.toFixed(2)
                        }
                    }
                }

                Column {

                    RowLayout {
                        Label {
                            text: "Attention Score"
                        }

                        Slider {

                            id: attentionSlider

                            from: 0
                            to: 1
                            value: 1

                            onPressedChanged: {
                                if (!pressed) {
                                    vehicleBridge.attentionChanged(value)
                                }
                            }
                        }

                        Label {
                            text: attentionSlider.value.toFixed(2)
                        }
                    }
                }
            }
 
        }

        GroupBox {
            id: trafficGroupBox
            title: "Traffic"

            Layout.fillWidth: true

            Column {
                ButtonGroup {
                    id: trafficGroup

                    onCheckedButtonChanged: {
                        if (trafficGroup.checkedButton) {
                            vehicleBridge.trafficChanged(trafficGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "No traffic"
                    ButtonGroup.group: trafficGroup
                    checked: true
                }
                RadioButton {
                    text: "Light"
                    ButtonGroup.group: trafficGroup
                }
                RadioButton {
                    text: "Heavy"
                    ButtonGroup.group: trafficGroup
                }
            }
        }

        GroupBox {
            id: roadTypeGroupBox
            title: "Road Type"

            Layout.fillWidth: true

            Column {
                ButtonGroup {
                    id: roadTypeGroup

                    onCheckedButtonChanged: {
                        if (roadTypeGroup.checkedButton) {
                            vehicleBridge.roadTypeChanged(roadTypeGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "Rural"
                    ButtonGroup.group: roadTypeGroup
                }
                RadioButton {
                    text: "Urban"
                    ButtonGroup.group: roadTypeGroup
                    checked: true
                }
                RadioButton {
                    text: "Highway"
                    ButtonGroup.group: roadTypeGroup
                }
            }
        }

        GroupBox {
            id: timeOfDayGroupBox
            title: "Time of Day"

            Layout.fillWidth: true

            Column {
                ButtonGroup {
                    id: timeOfDayGroup

                    onCheckedButtonChanged: {
                        if (timeOfDayGroup.checkedButton) {
                            vehicleBridge.timeOfDayChanged(timeOfDayGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "Morning"
                    ButtonGroup.group: timeOfDayGroup
                    checked: true
                }
                RadioButton {
                    text: "Afternoon"
                    ButtonGroup.group: timeOfDayGroup
                }
                RadioButton {
                    text: "Evening"
                    ButtonGroup.group: timeOfDayGroup
                }
                RadioButton {
                    text: "Night"
                    ButtonGroup.group: timeOfDayGroup
                }
            }
        }

        GroupBox {
            id: weatherGroupBox
            title: "Weather"

            Layout.fillWidth: true

            Column {
                ButtonGroup {
                    id: weatherGroup

                    onCheckedButtonChanged: {
                        if (weatherGroup.checkedButton) {
                            vehicleBridge.weatherChanged(weatherGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "Sunny"
                    ButtonGroup.group: weatherGroup
                    checked: true
                }
                RadioButton {
                    text: "Rainy"
                    ButtonGroup.group: weatherGroup
                }
                RadioButton {
                    text: "Foggy"
                    ButtonGroup.group: weatherGroup
                }
                RadioButton {
                    text: "Snowy"
                    ButtonGroup.group: weatherGroup
                }
            }
        }

        GroupBox {
            id: peopleAroundGroupBox
            title: "People Around"

            Layout.fillWidth: true

            Column {

                Slider {
                    id: peopleSlider
                    from: 0
                    to: 10
                    value: 0
                    stepSize: 1
                    snapMode: Slider.SnapAlways
                    onPressedChanged: {
                        if (!pressed) {
                            vehicleBridge.peopleAroundChanged(value)
                        }
                    }
                }

                Label {
                    text: Math.round(peopleSlider.value)
                }
            }
        }

        GroupBox {
            id: detectedObjectsGroupBox
            title: "Detected Objects"

            Layout.fillWidth: true

            Column {
                ButtonGroup {
                    id: detectedObjectsGroup

                    onCheckedButtonChanged: {
                        if (detectedObjectsGroup.checkedButton) {
                            vehicleBridge.detectedObjectsChanged(detectedObjectsGroup.checkedButton.text)
                        }
                    }
                }
                RadioButton {
                    text: "No objects"
                    ButtonGroup.group: detectedObjectsGroup
                    checked: true
                }
                RadioButton {
                    text: "Luggage"
                    ButtonGroup.group: detectedObjectsGroup
                }
                RadioButton {
                    text: "Gun"
                    ButtonGroup.group: detectedObjectsGroup
                }
            }
        }

        GroupBox {
            id: speedGroupBox
            title: "Speed"
            enabled: engineSwitch.checked

            Layout.fillWidth: true

            Column {

                Slider {

                    id: speedSlider

                    from: 0
                    to: 200
                    value: 0
                    onPressedChanged: {
                        if (!pressed) {
                            vehicleBridge.speedChanged(value)
                        }
                    }
                }

                Label {
                    text: Math.round(speedSlider.value) + " km/h"
                }
            }
        }

        GroupBox {
            id: engineGroupBox
            title: "Engine"

            Layout.fillWidth: true

            Switch {
                id: engineSwitch
                text: checked ? "ON" : "OFF"

                onCheckedChanged: {
                    vehicleBridge.engineChanged(checked)
                }
            }
        }

        GroupBox {
            id: doorsGroupBox
            title: "Doors"

            Layout.fillWidth: true

            Switch {

                text: checked ? "Unlocked" : "Locked"

                onCheckedChanged: {
                    vehicleBridge.doorsChanged(checked)
                }
            }
        }

        GroupBox {

            title: "Internal Temperature"

            Layout.fillWidth: true

            Column {

                Slider {

                    id: tempSlider

                    from: 10
                    to: 40
                    snapMode: Slider.SnapAlways
                    stepSize: 0.5
                    value: 22

                    onPressedChanged: {
                        if (!pressed) {
                            vehicleBridge.temperatureChanged(value)
                        }
                    }
                }

                Label {
                    text: tempSlider.value.toFixed(1) + " °C"
                }
            }
        }

        GroupBox {

            title: "External Temperature"

            Layout.fillWidth: true

            Column {

                Slider {

                    id: externalTempSlider

                    from: 10
                    to: 40
                    snapMode: Slider.SnapAlways
                    stepSize: 0.5
                    value: 22

                    onPressedChanged: {
                        if (!pressed) {
                            vehicleBridge.externalTemperatureChanged(value)
                        }
                    }
                }

                Label {
                    text: externalTempSlider.value.toFixed(1) + " °C"
                }
            }
        }

        GroupBox {
            id: riskyAreaGroupBox
            title: "Risky area"

            Layout.fillWidth: true

            Switch {

                text: checked ? "ON" : "OFF"

                onCheckedChanged: {
                    vehicleBridge.riskyAreaChanged(checked)
                }
            }
        }

        GroupBox {
            id: assistantGroupBox
            title: "Assistant"

            Layout.columnSpan: parent.columns
            Layout.fillWidth: true

            RowLayout {

                anchors.fill: parent

                TextArea {
                    id: inputField

                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    placeholderText: "Ask something..."

                    Keys.onReturnPressed: {
                        vehicleBridge.userInput(text)
                        clear()
                    }
                }

                Button {
                    text: "Send"

                    Layout.preferredWidth: 120
                    Layout.fillHeight: true

                    onClicked: {
                        vehicleBridge.userInput(inputField.text)
                        inputField.clear()
                    }
                }
            }
        }
    
    }
}