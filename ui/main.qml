import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1800
    height: 1000
    visible: true
    minimumWidth: 1800
    minimumHeight: 1000
    title: "Automotive AI Dashboard"

    readonly property bool isDark: vehicleBridge ? vehicleBridge.isDarkMode : false

    palette.window: isDark ? "#1e1e1e" : "#f5f5f7"
    palette.windowText: isDark ? "#ffffff" : "#1a1a1a"
    palette.base: isDark ? "#2b2b2b" : "#ffffff"
    palette.text: isDark ? "#ffffff" : "#1a1a1a"
    palette.button: isDark ? "#3a3a3c" : "#e5e5ea"
    palette.buttonText: isDark ? "#ffffff" : "#1a1a1a"
    palette.highlight: "#0a84ff"
    palette.highlightedText: "#ffffff"
    palette.mid: isDark ? "#48484a" : "#d1d1d6"

    // Simulates a gradually changing sensor: instead of sending a single
    // jump when a slider is released, sends several interpolated points
    // between the drag's start and end value, spaced out over time.
    function sendProgressive(fn, fromValue, toValue, steps) {
        steps = steps || 5
        if (fromValue === toValue) {
            fn(toValue)
            return
        }
        // Each call gets its own independent Timer (instead of a shared
        // queue/timer) so a slow fn() call on one tick can never cause
        // ticks to be dropped or interfere with another in-flight call.
        var timer = Qt.createQmlObject(
            "import QtQuick; Timer { interval: 150; repeat: true; running: true }",
            root,
            "progressiveTimer"
        )
        var count = 0
        timer.triggered.connect(function() {
            count++
            var v = fromValue + (toValue - fromValue) * count / steps
            fn(v)
            if (count >= steps) {
                timer.stop()
                timer.destroy()
            }
        })
    }

    Row {
        width: parent.width
        height: parent.height
        leftPadding: 10
        rightPadding: 10
        topPadding: 10
        bottomPadding: 10
        spacing: 10

        Column {
            width: parent.width*0.7-leftPadding-rightPadding-spacing
            height: parent.height-topPadding-bottomPadding-spacing
            spacing: parent.spacing

            Row {
                width: parent.width
                height: parent.height*0.02

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
                title: "Driver State"

                Row {
                    id: driverStateRow
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: parent.height
                    spacing: 10

                    GroupBox {
                        width: parent.width/3-spacing
                        height: parent.height
                        title: "Driver Activity"

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: activityGroup

                                onCheckedButtonChanged: {
                                    if (activityGroup.checkedButton) {
                                        vehicleBridge.stringChanged("DriverPhysicalState", "activity", ""+activityGroup.checkedButton.text)
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
                        title: "Driver Emotion State"
                        width: parent.width/3-spacing
                        height: parent.height

                        Column {
                            anchors.fill: parent
                            spacing: 10

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Angry"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: angrySlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "angry", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: angrySlider.value.toFixed(2)
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Disgust"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: disgustSlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "disgust", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: disgustSlider.value.toFixed(2)
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Fear"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: fearSlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "fear", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: fearSlider.value.toFixed(2)
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Happy"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: happySlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "happy", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: happySlider.value.toFixed(2)
                                }
                            }
                            
                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Sad"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: sadSlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "sad", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: sadSlider.value.toFixed(2)
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Surprise"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: surpriseSlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "surprise", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: surpriseSlider.value.toFixed(2)
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Neutral"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: neutralSlider
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 1

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverEmotionState", "neutral", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: neutralSlider.value.toFixed(2)
                                }
                            }
                        
                        }
                    }

                    GroupBox {
                        width: parent.width/3-spacing
                        height: parent.height
                        title: "Other"
                        
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
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverPhysicalState", "fatigue_level", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
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
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 1

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverPhysicalState", "attention_level", v) }, dragStartValue, value)
                                        }
                                    }
                                }

                                Label {
                                    width: parent.width*0.2
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
                                    width: parent.width*0.5
                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("DriverDrivingStyle", "aggressiveness_level", v) }, dragStartValue, value)
                                        }
                                    }
                                }

                                Label {
                                    width: parent.width*0.2
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
                title: "Environmental State"

                Row {
                    id: environmentStateRow
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: parent.height
                    spacing: 10

                    property real groupColSizePerc: 0.11

                    GroupBox {
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height
                        title: "Weather"

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: weatherGroup

                                onCheckedButtonChanged: {
                                    if (weatherGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","weather", weatherGroup.checkedButton.text)
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
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height
                        title: "Traffic"

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: trafficGroup

                                onCheckedButtonChanged: {
                                    if (trafficGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","traffic", trafficGroup.checkedButton.text)
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
                                text: "Medium"
                                ButtonGroup.group: trafficGroup
                            }
                            RadioButton {
                                text: "Heavy"
                                ButtonGroup.group: trafficGroup
                            }
                        }
                    }

                    GroupBox {
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height
                        title: "Road Type"

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: roadTypeGroup

                                onCheckedButtonChanged: {
                                    if (roadTypeGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","road_type", roadTypeGroup.checkedButton.text)
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
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height
                        title: "Risk Level"

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: riskLevelGroup

                                onCheckedButtonChanged: {
                                    if (riskLevelGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","risk_level", riskLevelGroup.checkedButton.text)
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
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height

                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: timeOfDayGroup

                                onCheckedButtonChanged: {
                                    if (timeOfDayGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","time_of_day", timeOfDayGroup.checkedButton.text)
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
                        width: parent.width*environmentStateRow.groupColSizePerc-environmentStateRow.spacing
                        height: parent.height
                        
                        Column {
                            anchors.fill: parent

                            ButtonGroup {
                                id: roadConditionGroup

                                onCheckedButtonChanged: {
                                    if (roadConditionGroup.checkedButton) {
                                        vehicleBridge.stringChanged("EnvironmentState","road_condition", roadConditionGroup.checkedButton.text)
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
                        width: parent.width*(1.0-environmentStateRow.groupColSizePerc*6)-environmentStateRow.spacing
                        height: parent.height
                        title: "Other"
                        
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
                                    width: parent.width*0.5

                                    from: 0
                                    to: 1
                                    value: 0

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("EnvironmentState","visibility", v) }, dragStartValue, value)
                                        }
                                    }

                                }

                                Label {
                                    width: parent.width*0.2
                                    text: visibilitySlider.value + " %"
                                }
                            }

                            Row {
                                width: parent.width
                                spacing: 10

                                Label {
                                    text: "Temperature (extl)"
                                    width: parent.width*0.3
                                }

                                Slider {
                                    id: externalTemperatureSlider
                                    width: parent.width*0.5

                                    from: 10
                                    to: 40
                                    snapMode: Slider.SnapAlways
                                    stepSize: 0.5
                                    value: 22

                                    property real dragStartValue: value

                                    onPressedChanged: {
                                        if (pressed) {
                                            dragStartValue = value
                                        } else {
                                            root.sendProgressive(function(v) { vehicleBridge.floatChanged("EnvironmentState","external_temperature", v) }, dragStartValue, value)
                                        }
                                    }
                                }

                                Label {
                                    width: parent.width*0.2
                                    text: externalTemperatureSlider.value.toFixed(1) + " °C"
                                }
                            }
                        }
                    }

                }
            }

            Row {
                width: parent.width
                height: parent.height*0.18-spacing
                spacing: 10

                GroupBox {
                    width: parent.width/3-spacing
                    height: parent.height
                    title: "Vehicle State"

                    Column {
                        id: vehicleStateRow
                        anchors.margins: 10
                        width: parent.width
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
                                    vehicleBridge.boolChanged("VehicleState", "engine_on", checked)
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
                                    vehicleBridge.boolChanged("VehicleState", "doors_unlocked", checked)
                                }
                            }
                        }

                        Row {
                            width: parent.width
                            spacing: 10

                            Label {
                                text: "Temperature (intl)"
                                width: parent.width*0.3
                            }

                            Slider {

                                id: tempSlider
                                width: parent.width*0.5

                                from: 10
                                to: 40
                                snapMode: Slider.SnapAlways
                                stepSize: 0.5
                                value: 22

                                property real dragStartValue: value

                                onPressedChanged: {
                                    if (pressed) {
                                        dragStartValue = value
                                    } else {
                                        root.sendProgressive(function(v) { vehicleBridge.floatChanged("VehicleState", "internal_temperature", v) }, dragStartValue, value)
                                    }
                                }
                            }

                            Label {
                                width: parent.width*0.2
                                text: tempSlider.value.toFixed(1) + " °C"
                            }
                        }

                    }
                }

                GroupBox {
                    width: parent.width/3-spacing
                    height: parent.height
                    title: "VehicleMotion"

                    Column {
                        anchors.margins: 10
                        height: parent.height
                        width: parent.width
                        spacing: 10

                        Row {
                            width: parent.width
                            height: parent.height
                            spacing: 10

                            Label {
                                text: "Speed"
                                width: parent.width*0.3
                            }

                            Slider {

                                id: speedSlider
                                width: parent.width*0.5

                                from: 0
                                to: 200
                                snapMode: Slider.SnapAlways
                                stepSize: 1
                                value: 0

                                property real dragStartValue: value

                                onPressedChanged: {
                                    if (pressed) {
                                        dragStartValue = value
                                    } else {
                                        root.sendProgressive(function(v) { vehicleBridge.floatChanged("VehicleMotion", "speed", v) }, dragStartValue, value)
                                    }
                                }
                            }

                            Label {
                                width: parent.width*0.2
                                text: speedSlider.value.toFixed(1) + " km/h"
                            }
                        }

                    }
                }

                GroupBox {
                    width: parent.width/3-spacing
                    height: parent.height
                    title: "DetectedObjects"

                    Column {
                        width: parent.width
                        anchors.margins: 10
                        height: parent.height
                        spacing: 10

                        Row {
                            width: parent.width
                            spacing: 10

                            Text {
                                text: "People Around"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: peopleSlider
                                width: parent.width*0.5
                                from: 0
                                to: 20
                                value: 0
                                stepSize: 1
                                snapMode: Slider.SnapAlways

                                property real dragStartValue: value

                                onPressedChanged: {
                                    if (pressed) {
                                        dragStartValue = value
                                    } else {
                                        root.sendProgressive(function(v) { vehicleBridge.intChanged("DetectedObjects", "people_around", Math.round(v)) }, dragStartValue, value)
                                    }
                                }
                            }

                            Label {
                                width: parent.width*0.2
                                text: Math.round(peopleSlider.value)
                            }
                        }

                        Row {
                            width: parent.width
                            spacing: 10

                            Text {
                                text: "Vehicles Around"
                                width: parent.width*0.3
                            }

                            Slider {
                                id: vehicleSlider
                                width: parent.width*0.5
                                from: 0
                                to: 20
                                value: 0
                                stepSize: 1
                                snapMode: Slider.SnapAlways

                                property real dragStartValue: value

                                onPressedChanged: {
                                    if (pressed) {
                                        dragStartValue = value
                                    } else {
                                        root.sendProgressive(function(v) { vehicleBridge.intChanged("DetectedObjects", "vehicles_around", Math.round(v)) }, dragStartValue, value)
                                    }
                                }
                            }

                            Label {
                                width: parent.width*0.2
                                text: Math.round(vehicleSlider.value)
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
                    spacing: 10

                    TextArea {
                        id: inputField
                        width: parent.width*0.75
                        height: parent.height

                        placeholderText: "Ask something..."

                        Keys.onReturnPressed: {
                            vehicleBridge.userInput(text)
                            clear()
                        }
                    }

                    Button {
                        text: "Send"
                        width: parent.width*0.1-parent.spacing
                        height: parent.height

                        onClicked: {
                            vehicleBridge.userInput(inputField.text)
                            inputField.clear()
                        }
                    }

                    Button {
                        id: conversationButton
                        checkable: true
                        text: checked ? "🎙️ Conversation ON" : "🎙️ Conversation"
                        width: parent.width*0.15-parent.spacing
                        height: parent.height

                        onCheckedChanged: {
                            if (checked) {
                                vehicleBridge.startConversation()
                            } else {
                                vehicleBridge.stopConversation()
                            }
                        }

                        Connections {
                            target: vehicleBridge
                            function onConversationModeChanged(active) {
                                conversationButton.checked = active
                            }
                        }
                    }
                }
            }

            Row {

                width: parent.width
                height: parent.height*0.05
                spacing: 10

                TextArea {
                    id: responseField
                    width: parent.width
                    height: parent.height
                    readOnly: true
                    placeholderText: "Agent response..."

                    Timer {
                        id: resetReponseFiled
                        interval: 5000
                        repeat: false
                        running: false
                        onTriggered: {
                            responseField.clear()
                        }
                    }

                    Connections {
                        target: vehicleBridge
                        function onResponseReceived(message) {
                            responseField.text = message
                            resetReponseFiled.start()
                        }
                    }
                }
            }
        }

        Column {
            width: parent.width*0.3-leftPadding-rightPadding-spacing*2
            height: parent.height-topPadding-bottomPadding-spacing*2
            spacing: parent.spacing

            GroupBox {
                width: parent.width
                height: parent.height
                title: "Knowledge"

                Connections {
                    target: vehicleBridge
                    function onKnowledgeUpdated(text) {
                        knowledgeTextArea.applyKnowledge(text)
                    }
                }

                ScrollView {
                    id: knowledgeScrollView
                    width: parent.width
                    height: parent.height
                    clip: true

                    ScrollBar.vertical.policy: ScrollBar.AsNeeded
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    TextArea {
                        id: knowledgeTextArea
                        width: knowledgeScrollView.availableWidth
                        readOnly: true
                        wrapMode: TextArea.WordWrap
                        text: "updating knowledge..."
                        font.family: "Courier New"
                        font.pixelSize: 12
                        selectByMouse: true

                        function applyKnowledge(updatedText) {
                            var flickable = knowledgeScrollView.contentItem
                            if (!flickable) {
                                if (text !== updatedText) {
                                    text = updatedText
                                }
                                return
                            }

                            var oldMaximum = Math.max(0, flickable.contentHeight - flickable.height)
                            var wasAtBottom = oldMaximum - flickable.contentY < 4
                            var savedContentY = flickable.contentY
                            if (text !== updatedText) {
                                text = updatedText
                            }

                            Qt.callLater(function() {
                                var newMaximum = Math.max(0, flickable.contentHeight - flickable.height)
                                flickable.contentY = wasAtBottom
                                    ? newMaximum
                                    : Math.min(savedContentY, newMaximum)
                            })
                        }
                    }
                }
                
            }
        }

    }

}