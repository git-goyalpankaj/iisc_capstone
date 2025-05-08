# app/memory/memory_store.py
from collections import deque
from typing import List, Dict

class MemoryStore:
    def __init__(self, max_turns: int = 5):
        # key: session_id → deque of turns
        self.store: Dict[str, deque] = {}
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.store:
            self.store[session_id] = deque(maxlen=self.max_turns)
        return list(self.store[session_id])

    def append_turn(self, session_id: str, turn: Dict):
        if session_id not in self.store:
            self.store[session_id] = deque(maxlen=self.max_turns)
        self.store[session_id].append(turn)
