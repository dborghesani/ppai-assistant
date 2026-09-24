from typing import Any
from datetime import datetime

class CarEvent:
    skill: str
    event_name: str
    event_value: Any
    context: list[str]
    timestamp: datetime
    user_input: str

    def __init__(self, skill: str, event_name: str, event_value: Any, 
                 context: list[str],
                 user_input: str = "",
                 skill_scope: str | None = None):
        self.skill = skill
        self.skill_scope = skill_scope
        self.event_name = event_name
        self.event_value = event_value
        self.context = list(context)
        self.user_input = user_input
        self.timestamp = datetime.now()

    