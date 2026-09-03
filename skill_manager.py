
from enum import Enum
import enum


class SkillType(Enum):
    CONVERSATION = "conversation"
    DRIVER_HEALTH = "driver_health"
    NAVIGATION = "navigation"
    PROACTIVE_SUGGESTIONS = "proactive_suggestions"
    VEHICLE = "vehicle"

class SkillManager:

    def __init__(self):
        self.skill_map: dict[SkillType, str] = {}
        for skill in enum.EnumMeta.__iter__(SkillType):
            self.skill_map[skill] = self.load(skill)
        
    def load(self, skill_name: SkillType) -> str:
        path = f"skills/{skill_name.value}.md"
        with open(path) as f:
            return f.read()

    def get_skill(self, skill_name: SkillType) -> str:
        return self.skill_map.get(skill_name, "no skill")

