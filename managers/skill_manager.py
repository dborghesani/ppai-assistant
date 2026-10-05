from pathlib import Path

from data.agents_dataclasses import SkillType

class SkillManager:

    def __init__(self):
        self.skill_map: dict[SkillType, str] = {}
        for skill in SkillType:
            self.skill_map[skill] = self.load(skill)
        action_guide_path = (
            Path(__file__).resolve().parents[1] / "skills" / "vehicle_actions.md"
        )
        self.vehicle_action_guidance = action_guide_path.read_text(encoding="utf-8")
        
    def load(self, skill_name: SkillType) -> str:
        if skill_name is SkillType.NONE:
            return ""
        path = f"skills/{skill_name.value}.md"
        with open(path) as f:
            return f.read()

    def get_skill(self, skill_name: SkillType) -> str:
        return self.skill_map.get(skill_name, "no skill")

