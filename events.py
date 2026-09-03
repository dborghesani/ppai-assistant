from typing import Any
from datetime import datetime

class CarEvent:
    skill: str
    event_name: str
    event_value: Any
    context: Any
    previous_contexts: list[Any] = []
    timestamp: datetime
    user_input: str = ""
    recent_notifications: list[Any] = []

    def __init__(self, skill: str, event_name: str, event_value: Any, 
                 context: Any, previous_contexts: list[Any] = [], user_input: str = ""):
        self.skill = skill
        self.event_name = event_name
        self.event_value = event_value
        self.context = context
        self.previous_contexts = previous_contexts
        self.user_input = user_input
        self.timestamp = datetime.now()

    