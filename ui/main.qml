import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1400
    height: 900
    visible: true
    minimumWidth: 1400
    minimumHeight: 900
    title: "Automotive AI Dashboard"

    Column {
        width: parent.width-leftPadding-rightPadding
        height: parent.height
        leftPadding: 10
        rightPadding: 10
        spacing: 10

        Row {
            width: parent.width
            height: parent.height*0.05

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

                Timer {
                    interval: 1500
                    repeat: false
                    running: true
                    onTriggered: {
                        eventProcessingSwitch.checked = true
                    }
                }
            }
            
        }

        GroupBox {
            width: parent.width
            height: parent.height*0.3
            title: "DriverState"

            Row {
                id: driverStateRow
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 10
                height: parent.height
                spacing: 10

                GroupBox {
                    width: parent.width*0.33
                    height: parent.height
                    title: "activity"

                    Column {
                        anchors.fill: parent

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
                    title: "mood"
                    width: parent.width*0.33
                    height: parent.height

                    Column {
                        anchors.fill: parent

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
                    width: parent.width*0.33
                    height: parent.height
                    title: " "
                    
                    Column {
                        width: parent.width
                        height: parent.height
                        spacing: 10

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                text: "Fatigue"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: fatigueSlider
                                width: parent.width*0.4

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
                                width: parent.width*0.3
                                text: fatigueSlider.value.toFixed(2)
                            }
                        }

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                text: "Attention"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: attentionSlider
                                width: parent.width*0.4

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
                                width: parent.width*0.3
                                text: attentionSlider.value.toFixed(2)
                            }
                        }

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                width: parent.width*0.3
                                text: "Aggressiveness"
                            }

                            Slider {
                                id: aggressivenessSlider
                                width: parent.width*0.4
                                from: 0
                                to: 1
                                value: 0

                                onPressedChanged: {
                                    if (!pressed) {
                                        vehicleBridge.aggressivenessChanged(value)
                                    }
                                }
                            }

                            Label {
                                width: parent.width*0.3
                                text: aggressivenessSlider.value.toFixed(2)
                            }
                        }
                    }
                }
            }
        }

        GroupBox {
            width: parent.width
            height: parent.height*0.3
            title: "EnvironmentState"

            Row {
                id: environmentStateRow
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 10
                height: parent.height
                spacing: 10

                property real groupColSizePerc: 0.12

                GroupBox {
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height
                    title: "weather"

                    Column {
                        anchors.fill: parent

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
                            text: "Cloudy"
                            ButtonGroup.group: weatherGroup
                        }
                        RadioButton {
                            text: "Rainy"
                            ButtonGroup.group: weatherGroup
                        }
                        RadioButton {
                            text: "Snowy"
                            ButtonGroup.group: weatherGroup
                        }
                        RadioButton {
                            text: "Foggy"
                            ButtonGroup.group: weatherGroup
                        }
                    }
                }

                GroupBox {
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height
                    title: "traffic"

                    Column {
                        anchors.fill: parent

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
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height
                    title: "road type"

                    Column {
                        anchors.fill: parent

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
                        RadioButton {
                            text: "Residential"
                            ButtonGroup.group: roadTypeGroup
                        }
                    }
                }

                GroupBox {
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height
                    title: "risk level"

                    Column {
                        anchors.fill: parent

                        ButtonGroup {
                            id: riskLevelGroup

                            onCheckedButtonChanged: {
                                if (riskLevelGroup.checkedButton) {
                                    vehicleBridge.riskLevelChanged(riskLevelGroup.checkedButton.text)
                                }
                            }
                        }

                        RadioButton {
                            text: "Low"
                            ButtonGroup.group: riskLevelGroup
                            checked: true
                        }
                        RadioButton {
                            text: "Medium"
                            ButtonGroup.group: riskLevelGroup
                        }
                        RadioButton {
                            text: "High"
                            ButtonGroup.group: riskLevelGroup
                        }
                    }
                }

                GroupBox {
                    id: timeOfDayGroupBox
                    title: "Time of Day"
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height

                    Column {
                        anchors.fill: parent

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
                    id: roadCondition
                    title: "Road Condition"
                    width: parent.width*environmentStateRow.groupColSizePerc
                    height: parent.height
                    Column {
                        anchors.fill: parent

                        ButtonGroup {
                            id: roadConditionGroup

                            onCheckedButtonChanged: {
                                if (roadConditionGroup.checkedButton) {
                                    vehicleBridge.roadConditionChanged(roadConditionGroup.checkedButton.text)
                                }
                            }
                        }

                        RadioButton {
                            text: "Dry"
                            ButtonGroup.group: roadConditionGroup
                            checked: true
                        }
                        RadioButton {
                            text: "Wet"
                            ButtonGroup.group: roadConditionGroup
                        }
                        RadioButton {
                            text: "Icy"
                            ButtonGroup.group: roadConditionGroup
                        }
                        RadioButton {
                            text: "Gravel"
                            ButtonGroup.group: roadConditionGroup
                        }
                        RadioButton {
                            text: "Snowy"
                            ButtonGroup.group: roadConditionGroup
                        }
                        RadioButton {
                            text: "Muddy"
                            ButtonGroup.group: roadConditionGroup
                        }
                    }
                }
                
                GroupBox {
                    width: parent.width*0.23
                    height: parent.height
                    title: " "
                    
                    Column {
                        width: parent.width
                        height: parent.height
                        spacing: 10

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                text: "Visibility"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: visibilitySlider
                                width: parent.width*0.4

                                from: 0
                                to: 1
                                value: 0

                                onPressedChanged: {
                                    if (!pressed) {
                                        vehicleBridge.visibilityChanged(value)
                                    }
                                }

                            }

                            Label {
                                width: parent.width*0.3
                                text: visibilitySlider.value + " %"
                            }
                        }

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                text: "Temperature"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: externalTemperatureSlider
                                width: parent.width*0.4

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
                                width: parent.width*0.3
                                text: externalTemperatureSlider.value.toFixed(1) + " °C"
                            }
                        }
                    }
                }

            }
        }

        Row {
            width: parent.width
            height: parent.height*0.2
            spacing: 10

            GroupBox {
                width: parent.width*0.33
                height: parent.height
                title: "VehicleState"

                Column {
                    id: vehicleStateRow
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: 10
                    height: parent.height
                    spacing: 10

                    Row {
                        id: engineRow
                        width: parent.width

                        Label {
                            text: "Engine"
                            width: parent.width*0.3
                        }

                        Switch {
                            id: engineSwitch
                            width: parent.width*0.7
                            text: checked ? "ON" : "OFF"

                            onCheckedChanged: {
                                vehicleBridge.engineChanged(checked)
                            }
                        }

                    }

                    Row {
                        id: doorsRow
                        width: parent.width

                        Label {
                            text: "Doors"
                            width: parent.width*0.3
                        }

                        Switch {
                            id: doorsSwitch
                            width: parent.width*0.7
                            text: checked ? "Unlocked" : "Locked"

                            onCheckedChanged: {
                                vehicleBridge.doorsChanged(checked)
                            }
                        }
                    }

                    Row {
                        width: parent.width
                        spacing: 10

                        Label {
                            text: "Temperature"
                            width: parent.width*0.3
                        }

                        Slider {

                            id: tempSlider
                            width: parent.width*0.4

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
                            width: parent.width*0.3
                            text: tempSlider.value.toFixed(1) + " °C"
                        }
                    }

                }
            }

            GroupBox {
                width: parent.width*0.33
                height: parent.height
                title: "VehicleMotion"

                Column {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: 10
                    height: parent.height
                    spacing: 10

                    Row {
                        width: parent.width
                        spacing: 10

                        Label {
                            text: "Speed"
                            width: parent.width*0.3
                        }

                        Slider {

                            id: speedSlider
                            width: parent.width*0.4

                            from: 0
                            to: 200
                            snapMode: Slider.SnapAlways
                            stepSize: 1
                            value: 0

                            onPressedChanged: {
                                if (!pressed) {
                                    vehicleBridge.speedChanged(value)
                                }
                            }
                        }

                        Label {
                            width: parent.width*0.3
                            text: speedSlider.value.toFixed(1) + " km/h"
                        }
                    }

                }
            }

            GroupBox {
                width: parent.width*0.33
                height: parent.height
                title: "DetectedObjects"

                Column {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: 10
                    height: parent.height
                    spacing: 10

                    Row {
                        width: parent.width
                        spacing: 10

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

                    Row {
                        width: parent.width
                        spacing: 10

                        Slider {
                            id: carsSlider
                            from: 0
                            to: 10
                            value: 0
                            stepSize: 1
                            snapMode: Slider.SnapAlways
                            onPressedChanged: {
                                if (!pressed) {
                                    vehicleBridge.carsAroundChanged(value)
                                }
                            }
                        }

                        Label {
                            text: Math.round(carsSlider.value)
                        }
                        
                    }

                }
            }

        }
        
        GroupBox {
            id: assistantGroupBox
            title: "Assistant"
            width: parent.width
            height: parent.height*0.1
            spacing: 10

            Row {

                width: parent.width
                height: parent.height

                TextArea {
                    id: inputField
                    width: parent.width*0.9
                    height: parent.height

                    placeholderText: "Ask something..."

                    Keys.onReturnPressed: {
                        vehicleBridge.userInput(text)
                        clear()
                    }
                }

                Button {
                    text: "Send"
                    width: parent.width*0.1
                    height: parent.height

                    onClicked: {
                        vehicleBridge.userInput(inputField.text)
                        inputField.clear()
                    }
                }
            }
        }
    }

}