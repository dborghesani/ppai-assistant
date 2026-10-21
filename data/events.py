from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from data.agents_dataclasses import SkillType


class EventName(StrEnum):
    KNOWLEDGE_UPDATED = "knowledge_updated"
    USER_INPUT = "user_input"


@dataclass(frozen=True)
class IncomingMessage:
    sender: str
    text: str

@dataclass(eq=False)
class CarEvent:
    skill: SkillType
    event_name: EventName
    event_value: Any
    context: list[str]
    timestamp: datetime = field(init=False, default_factory=datetime.now)
    user_input: str = ""
    facts: dict[str, str] = field(default_factory=dict, kw_only=True)
    changed_facts: list[str] | None = field(default=None, kw_only=True)
    supporting_facts: list[str] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self.event_name = EventName(self.event_name)
        self.context = list(self.context)
        self.facts = dict(self.facts)
        if self.changed_facts is not None:
            self.changed_facts = list(self.changed_facts)
        if self.supporting_facts is not None:
            self.supporting_facts = list(self.supporting_facts)

        
