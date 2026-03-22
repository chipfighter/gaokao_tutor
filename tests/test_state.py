"""Unit tests for TutorState definition."""

from src.graph.state import TutorState


class TestTutorState:

    def test_state_has_required_keys(self):
        annotations = TutorState.__annotations__
        required = ["messages", "intent", "subject", "keypoints", "context", "plan"]
        for key in required:
            assert key in annotations, f"TutorState missing key: {key}"

    def test_state_instantiation(self):
        state: TutorState = {
            "messages": [],
            "intent": "academic",
            "subject": "math",
            "keypoints": [],
            "context": [],
            "plan": "",
        }
        assert state["intent"] == "academic"
        assert isinstance(state["messages"], list)

    def test_state_accepts_all_intents(self):
        for intent in ("academic", "planning", "emotional"):
            state: TutorState = {
                "messages": [],
                "intent": intent,
                "subject": "",
                "keypoints": [],
                "context": [],
                "plan": "",
            }
            assert state["intent"] == intent
