from agents.agents_dataclasses import SkillType

class SkillManager:

    def __init__(self):
        self.skill_map: dict[SkillType, str] = {}
        for skill in SkillType:
            self.skill_map[skill] = self.load(skill)
        
    def load(self, skill_name: SkillType) -> str:
        if skill_name is SkillType.NONE:
            return ""
        path = f"skills/{skill_name.value}.md"
        with open(path) as f:
            return f.read()

    def get_skill(self, skill_name: SkillType) -> str:
        return self.skill_map.get(skill_name, "no skill")

