"""Define the shared values."""
from dataclasses import dataclass
from langgraph.graph import MessagesState
from typing import List

@dataclass(kw_only=True)
class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]