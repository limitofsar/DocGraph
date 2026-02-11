from typing import TypedDict, List, Literal, Optional

class Docs(TypedDict):
    raw_text: str
    clf_text: str
    source: Literal["vector_db", "web"]
    url: Optional[str]
    score: Optional[float]

class State(TypedDict):
    question: str
    route: Literal['chat', 'retrieval']
    docs: List[Docs]
    source: Literal['vector_db', 'web', 'none']
    answer: str

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
