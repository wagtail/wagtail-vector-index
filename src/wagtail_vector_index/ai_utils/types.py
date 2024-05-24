from collections.abc import Callable
from typing import Any, Protocol, TypedDict


class ChatMessage(TypedDict):
    """Dict contaning 'role' and 'content' keys representing a single message in
    a chat conversation."""

    role: str
    content: str


class AIResponseStreamingPart(TypedDict):
    """Represents a part of a streaming AI response where the index represents the choice
    if multiple choices are returned by the backend."""

    index: int
    content: str


class AIStreamingResponse:
    """Iterator protocol representing a streaming response from an AI backend.

    Each iteration returns a single AIResponseStreamingPart. As a backend may return multiple
    'choices' in an undetermined order, the consumer must reconstruct the response based on the 'index'
    field of each part.

    e.g. This will return something something like:
    [
        {"index": 0, "content": "I"},
        {"index": 1, "content": "ChatGPT"},
        {"index": 1, "content": "is"},
        {"index": 1, "content": "an"},
        {"index": 0, "content": "am"},
        {"index": 0, "content": "ChatGPT"},
        {"index": 1, "content": "AI"},
    ]
    """

    def __iter__(self):
        return self

    def __next__(self) -> AIResponseStreamingPart: ...


class AIResponse:
    """Representation of a non-streaming response from an AI backend.

    `choices` is a list of strings representing the choices returned by the backend.
    """

    def __init__(self, choices: list[str]) -> None:
        self.choices = choices


class TextSplitterProtocol(Protocol):
    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        length_function: Callable[[str], int],
        **kwargs: Any,
    ) -> None: ...

    def split_text(self, text: str) -> list[str]: ...


class TextSplitterLengthCalculatorProtocol(Protocol):
    def get_splitter_length(self, text: str) -> int: ...
