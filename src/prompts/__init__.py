from src.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from src.prompts.academic import (
    ACADEMIC_SYSTEM_PROMPT,
    ACADEMIC_ANSWER_PROMPT,
)
from src.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_GENERATE_PROMPT,
)
from src.prompts.emotional import EMOTIONAL_SYSTEM_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "ACADEMIC_SYSTEM_PROMPT",
    "ACADEMIC_ANSWER_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_GENERATE_PROMPT",
    "EMOTIONAL_SYSTEM_PROMPT",
]
