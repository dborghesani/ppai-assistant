from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from data.agents_dataclasses import SkillType


class EventName(StrEnum):
    INCOMING_MESSAGE_RECEIVED = "incoming_message_received"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    PENDING_MESSAGES_REMINDER = "pending_messages_reminder"
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

    def __post_init__(self) -> None:
        self.event_name = EventName(self.event_name)
        self.context = list(self.context)

        
