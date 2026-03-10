from typing import List, Optional
import uuid


class Turn:
    def __init__(self, previous_response, current_prompt, index) -> None:
        self.previous_response: str = previous_response
        self.current_prompt: str = current_prompt
        self.index: int = index
        self.user_prompt_embedding: Optional[List[float]] = None
        self.previous_response_embedding: Optional[List[float]] = None
        self.combined_prompt: str = ""
        self.combined_prompt_embedding: Optional[List[float]] = None
        self.intent: str = ""
        self.intent_embedding: Optional[List[float]] = None


class Trajectory:
    def __init__(self) -> None:
        self.attack_id = str(uuid.uuid4())
        self.trajectory: List[Turn] = []
        self.success: bool = False
        self.payload_embedding: Optional[List[float]] = None
        self.payload_text: str = ""

    def __len__(self) -> int:
        return len(self.trajectory)

    def append(self, turn: Turn) -> None:
        self.trajectory.append(turn)
