from dataclasses import dataclass, field
from typing import Any
import quaternion


def knowledge_field(
    *,
    change_threshold: float | None = None,
    change_ratio: float | None = None,
    notify_on_trend_change: bool = True,
) -> Any:
    return field(default=None, metadata={
        "knowledge": True,
        "change_threshold": change_threshold,
        "change_ratio": change_ratio,
        "notify_on_trend_change": notify_on_trend_change,
    })


@dataclass
class GPSData:
    latitude: float | None = None  # degrees
    latitudeHemisphere: str | None = None  # N or S
    longitude: float | None = None  # degrees
    longitudeHemisphere: str | None = None  # E or W
    utcDate: str | None = None  # format: DDMMYYYY
    utcTime: int | None = None  # format HHMMSS in UTC time
    speed: float | None = None  # km/h
    course: float | None = None  # degrees
    magneticVariation: float | None = None  # unit unknown

@dataclass
class GPSIMUData:
    latitude: float | None = None  # degrees
    latitudeHemisphere: str | None = None  # N or S
    longitude: float | None = None  # degrees
    longitudeHemisphere: str | None = None  # E or W
    utcDate: str | None = None  # format: DDMMYYYY
    utcTime: int | None = None  # format HHMMSS in UTC time
    altitude: float | None = None  # m, GPS altitude
    speed: float | None = None  # km/h
    course: float | None = None  # degrees
    magnetic_field_x: float | None = None  # x, y, z (unit unknown)
    magnetic_field_y: float | None = None  # x, y, z (unit unknown)
    magnetic_field_z: float | None = None  # x, y, z (unit unknown)
    pressure_pa: float | None = None  # Pa
    altitude_m: float | None = None  # m, barometric altitude
    quat: quaternion.quaternion | None = None  # q0, q1, q2, q3

@dataclass 
class GPSIMUState:
    satellitesCount: int | None = None
    pdop: float | None = None
    hdop: float | None = None
    vdop: float | None = None

@dataclass
class VehicleMotion:
    speed: float | None = knowledge_field(change_threshold=5.0)  # km/h, aggregated
    wheel_speed_front_left: float | None = None  # km/h
    wheel_speed_front_right: float | None = None  # km/h
    wheel_speed_rear_left: float | None = None  # km/h
    wheel_speed_rear_right: float | None = None  # km/h
    yaw_speed: float | None = None  # deg/s
    acceleration_longitudinal: float | None = knowledge_field(change_threshold=0.5)  # m/s^2
    acceleration_lateral: float | None = knowledge_field(change_threshold=0.5)  # m/s^2
    steering_angle: float | None = knowledge_field(change_threshold=5.0)  # degrees
    engine_rpm: float | None = knowledge_field(change_threshold=250.0)  # rpm
    # from GPSIMU:
    acceleration_g_x: float | None = None  # g
    acceleration_g_y: float | None = None  # g
    acceleration_g_z: float | None = None  # g
    angular_velocity_dps_x: float | None = None  # deg/s
    angular_velocity_dps_y: float | None = None  # deg/s
    angular_velocity_dps_z: float | None = None  # deg/s
    orientation_degrees_yaw: float | None = None  # deg
    orientation_degrees_pitch: float | None = None  # deg
    orientation_degrees_roll: float | None = None  # deg

@dataclass
class VehicleState:
    door_open_front_left: bool | None = knowledge_field()
    door_open_front_right: bool | None = knowledge_field()
    door_open_rear_left: bool | None = knowledge_field()
    door_open_rear_right: bool | None = knowledge_field()
    doors_unlocked: bool | None = knowledge_field()
    lights_on_sidelights: bool | None = knowledge_field()
    lights_on_low_beams: bool | None = knowledge_field()
    lights_on_high_beams: bool | None = knowledge_field()
    lights_on_fog_lights: bool | None = knowledge_field()
    turn_signal: str | None = knowledge_field()  # ["Off", "Right", "Left", "Both"]
    trunk_open: bool | None = knowledge_field()
    lane_keep_assist: str | None = knowledge_field()  # ["Unavailable", "Unselected", "Selected", "Authorized", "Active", "Defect", "Collision_Risk_not_used_during_LPA"]
    blind_spot_monitor: bool | None = knowledge_field()
    engine_on: bool | None = knowledge_field()
    internal_temperature: float | None = knowledge_field(
        change_threshold=1.0,
        notify_on_trend_change=False,
    )  # °C

@dataclass
class LaneTrace:
    distance: float | None = None  # m
    heading_angle: float | None = None  # degrees
    C0: float | None = None  # same value as distance
    C1: float | None = None  # heading_angle converted to radians
    C2: float | None = None
    C3: float | None = None
    type: str | None = None  # ["Undetermined", "Solid line", "Dotted line", "Double line", "Botts Dots", "Road Edge", "Barrier or curb", "Invalid"]
    confidence: int | None = None  # between 0-10
    view_range_start: float | None = None  # m
    view_range_end: float | None = None  # m

@dataclass
class LaneTracing:
    road_type: str | None = None  # ["Unknown", "Highway", "Inner_city", "Interurban"]
    highway_exit: str | None = None  # ["No exit", "Right", "Left"]
    lane_crossing_left: bool | None = None
    lane_crossing_right: bool | None = None
    lane_L0: LaneTrace | None = None
    lane_L1: LaneTrace | None = None
    lane_R0: LaneTrace | None = None
    lane_R1: LaneTrace | None = None

    # Derived lane counters (computed using lane detection and lane types):
    total_lane: int | None = None
    driving_lane: int | None = None  # start at 1

@dataclass
class VisionObject:
    track_id: int | None = None
    category: str | None = None  # ["Unknown", "Truck", "Car", "Motor Bike", "Bicycle", "Pedestrian", "Undecided"]
    distance_longitudinal: float | None = None  # m
    distance_lateral: float | None = None  # m
    length: float | None = None  # m, default to 25.4 for pedestrian and bicycle
    yaw_angle: float | None = None  # rad
    heading: float | None = None  # yaw_angle converted to deg
    lane_position: str | None = None  # ["Not assigned", "Ego Lane", "Left Lane", "Right Lane"]
    longitudinal_relative_velocity: float | None = None  # m/s
    lateral_relative_velocity: float | None = None  # m/s
    reference_point: str | None = None  # ["Not Visible", "Front", "Rear"]
    left_edge_view: float | None = None  # Viewing angle of the vehicle's left vertical edge. The angle is in camera coordinates. The left edge angle shall specify the rear or front view of vehicles; for pedestrians these angles may also specify the side view.
    right_edge_view: float | None = None  # Viewing angle of the vehicle's right vertical edge. The angle is in camera coordinates. The right edge angle shall specify the rear or front view of vehicles; for pedestrians these angles may also specify the side view.
    distance_edge_view: float | None = None  # no description
    prob_class: float | None = None  # %, probability for the given type classification
    prob_existence: float | None = None  # %, probability that the object exists
    brake_light_status: str | None = None  # ["Not Available", "On", "Off"]
    turn_indicator_status: str | None = None  # ["Not Available", "Off", "Left", "Right", "Both"]

@dataclass
class RadarObject:
    track_id: int | None = None
    is_measured: bool | None = None
    is_measurable: bool | None = None
    yaw_angle: float | None = None  # rad
    length: float | None = None  # m
    width: float | None = None  # m
    distance_lateral: float | None = None  # m
    distance_longitudinal: float | None = None  # m; front radars FL/FR use front distance, rear radars RL/RR use rear distance
    lateral_velocity: float | None = None  # m/s
    longitudinal_velocity: float | None = None  # m/s
    lateral_acceleration: float | None = None  # m/s²
    longitudinal_acceleration: float | None = None  # m/s²
    radar_reference_point: str | None = None  # ["Front corner left", "Front center", "Front corner right", "Front wheel case left", "Front wheel case right", "Side middle left", "Side middle right", "Rear wheel case left", "Rear wheel case right", "Rear corner left", "Rear center", "Rear corner right", "Object center", "Rear wheel case center", "Front wheel case center", "Invalid"]
    prob_existence: float | None = None  # %, probability that the object exists
    prob_car: float | None = None  # %
    prob_truck: float | None = None  # %
    prob_2_wheels: float | None = None  # %
    prob_4_wheels: float | None = None  # %
    prob_pedestrian: float | None = None  # %
    prob_obstacle: float | None = None  # %
    prob_not_obstacle: float | None = None  # %
    prob_roadside_obj: float | None = None  # %
    prob_in_motion: float | None = None  # %
    prob_static: float | None = None  # %
    prob_instant_movement: float | None = None  # %
    prob_previous_movement: float | None = None  # %
    lca_status: str | None = None

@dataclass
class TrafficSign:
    type_id: int | None = None  # unknown map
    x_position: float | None = None  # m
    y_position: float | None = None  # m
    subsign_id: str | None = None  # ["None", "Unknown", "Time", "Distance Arrow", "Distance", "Remind"]

@dataclass
class TrafficSigns:
    speed_limit: float | None = None  # km/h
    sign_1: TrafficSign | None = field(default_factory=TrafficSign)
    sign_2: TrafficSign | None = field(default_factory=TrafficSign)
    sign_3: TrafficSign | None = field(default_factory=TrafficSign)
    sign_4: TrafficSign | None = field(default_factory=TrafficSign)

@dataclass
class DetectedObjects:
    vision_objects: list[VisionObject] | None = field(default_factory=list)
    radar_objects: list[RadarObject] | None = field(default_factory=list)
    traffic_signs: TrafficSigns | None = field(default_factory=TrafficSigns)
    people_around: int | None = knowledge_field(change_threshold=1.0)
    vehicles_around: int | None = knowledge_field(change_threshold=1.0)
    dangerous_objects_around: int | None = knowledge_field(change_threshold=1.0)

@dataclass
class DriverPhysicalState:
    activity: str | None = knowledge_field()  # ["Idle", "Driving", "Talking", "Using Phone", "Eating", "Sleeping"]
    attention_level: float | None = knowledge_field(change_threshold=0.1)  # [0.0, 1.0]
    fatigue_level: float | None = knowledge_field(change_threshold=0.1)  # [0.0, 1.0]

@dataclass
class DriverEmotionState:
    angry: float | None = knowledge_field(change_threshold=0.15)
    disgust: float | None = knowledge_field(change_threshold=0.15)
    fear: float | None = knowledge_field(change_threshold=0.15)
    happy: float | None = knowledge_field(change_threshold=0.15)
    sad: float | None = knowledge_field(change_threshold=0.15)
    surprise: float | None = knowledge_field(change_threshold=0.15)
    neutral: float | None = knowledge_field(change_threshold=0.15)

@dataclass
class DriverDrivingStyle:
    aggressiveness_level: float | None = knowledge_field(change_threshold=0.1)  # [0.0, 1.0] [safe, aggressive]
    
@dataclass
class EnvironmentState:
    external_temperature: float | None = knowledge_field(
        change_threshold=2.0,
        notify_on_trend_change=False,
    )  # °C
    weather: str | None = knowledge_field()  # ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
    time_of_day: str | None = knowledge_field()  # ["Morning", "Afternoon", "Evening", "Night"]
    road_condition: str | None = knowledge_field()  # ["Dry", "Wet", "Icy", "Snowy", "Gravel"]
    road_type: str | None = knowledge_field()  # ["Urban", "Rural", "Highway", "Residential"]
    risk_level: str | None = knowledge_field()  # ["Low", "Medium", "High"]
    visibility: str | None = knowledge_field()  # ["Clear", "Moderate", "Poor"]
    traffic: str | None = knowledge_field()  # ["No traffic", "Light", "Heavy"]

