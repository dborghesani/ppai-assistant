from skill_manager import SkillType

class SkillMap:
    
    def __init__(self):
        self.event_skill_map = {}
        self.event_skill_map["DriverState"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["EnvironmentState"] = [SkillType.PROACTIVE_SUGGESTIONS, SkillType.NAVIGATION, SkillType.DRIVER_HEALTH]
        self.event_skill_map["VehicleMotion"] = [SkillType.NAVIGATION]
        """
        self.event_skill_map["DriverState.behavior"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["DriverState.fatigue_level"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["DriverState.attention_level"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["DriverState.activity"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["EnvironmentState.weather"] = [SkillType.PROACTIVE_SUGGESTIONS, SkillType.NAVIGATION]
        #self.event_skill_map["external_temperature"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["EnvironmentState.people_around"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["speed"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["EnvironmentState.traffic"] = [SkillType.PROACTIVE_SUGGESTIONS, SkillType.NAVIGATION, SkillType.DRIVER_HEALTH]
        self.event_skill_map["road_type"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["time_of_day"] = [SkillType.PROACTIVE_SUGGESTIONS,SkillType.DRIVER_HEALTH]
        self.event_skill_map["detected_objects"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["risky_area"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["engine_running"] = [SkillType.VEHICLE]
        self.event_skill_map["doors_unlocked"] = [SkillType.VEHICLE]
        self.event_skill_map["internal_temperature"] = [SkillType.VEHICLE]
        self.event_skill_map["destination"] = [SkillType.NAVIGATION]
        self.event_skill_map["user_input"] = [SkillType.CONVERSATION]
        """

    def get_skills_for_event(self, event_type: str) -> list[str]:
        return self.event_skill_map.get(event_type, [])

    def get_all_event_types(self) -> list[str]:
        return list(self.event_skill_map.keys())

    def get_all_skills(self) -> list[str]:
        return list(set(skill for skills in self.event_skill_map.values() for skill in skills))

    def get_events_for_skill(self, skill: SkillType) -> list[str]:
        return [event for event, skills in self.event_skill_map.items() if skill in skills]