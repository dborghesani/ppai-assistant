from typing import List
from dataclasses import dataclass

from skill_manager import SkillType

@dataclass
class AssistantState:
    # TODO find a way of updating this and UI default
    driving_behavior: str = "Normal"
    driver_activity: str = "Idle"
    driver_mood: str = "Neutral"

    fatigue_score: float = 0.0
    attention_score: float = 1.0

    traffic: str = "No traffic"
    road_type: str = "Urban"
    weather: str = "Sunny"
    time_of_day: str = "Morning"

    people_around: int = 0
    detected_objects: str = "No objects"

    speed: int = 0
    engine_on: bool = False
    doors_unlocked: bool = True

    internal_temperature: int = 22
    external_temperature: int = 22

    risky_area: bool = False

class AssistantManager:
    event_skill_map: dict[str, List[SkillType]] = {}
    assistant_state: AssistantState = AssistantState()
    
    def __init__(self):
        self.event_skill_map = {}
        self.event_skill_map["driver_mood"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["driving_behavior"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["fatigue_score"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["attention_score"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["driver_activity"] = [SkillType.DRIVER_HEALTH]
        self.event_skill_map["weather"] = [SkillType.PROACTIVE_SUGGESTIONS, SkillType.NAVIGATION]
        self.event_skill_map["external_temperature"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["people_around"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["speed"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["traffic"] = [SkillType.PROACTIVE_SUGGESTIONS, SkillType.NAVIGATION]
        self.event_skill_map["road_type"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["time_of_day"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["detected_objects"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["risky_area"] = [SkillType.PROACTIVE_SUGGESTIONS]
        self.event_skill_map["engine_running"] = [SkillType.VEHICLE]
        self.event_skill_map["doors_unlocked"] = [SkillType.VEHICLE]
        self.event_skill_map["internal_temperature"] = [SkillType.VEHICLE]
        self.event_skill_map["destination"] = [SkillType.NAVIGATION]

    def get_skills_for_event(self, event_type: str) -> List[SkillType]:
        return self.event_skill_map.get(event_type, [])

    def get_all_event_types(self) -> List[str]:
        return list(self.event_skill_map.keys())

    def get_all_skills(self) -> List[SkillType]:
        return list(set(skill for skills in self.event_skill_map.values() for skill in skills))

    def get_events_for_skill(self, skill: SkillType) -> List[str]:
        return [event for event, skills in self.event_skill_map.items() if skill in skills]

    def get_context_for_event(self, event_type: str) -> dict:
        if event_type not in self.event_skill_map:
            return {}

        # get the skills associated to the event
        skills = self.event_skill_map[event_type]

        # build the context
        context = {}
        for skill in skills:
            events_for_context = self.get_events_for_skill(skill)
            for event in events_for_context:
                context[event] = getattr(self.assistant_state, event, None)
        return context

