from collections.abc import Callable
from typing import Any, Protocol, TypedDict


class ChatMessage(TypedDict):
    """Dict contaning 'role' and 'content' keys representing a single message in
    a chat conversation."""

    role: str
    content: str


class AIResponseStreamingPart(TypedDict):
    """When the AI backend streams a response, it may return multiple 'choices'
    in one stream. The chunks of the response may come in any order, so this structure
    allows the consumer to reconstruct each separate stream using the choice 'index'."""

    index: int
    content: str


class AIStreamingResponse:
    """Iterator protocol representing a streaming response from an AI backend."""

    def __iter__(self):
        return self

    def __next__(self) -> AIResponseStreamingPart: ...


class AIResponse:
    """Representation of a non-streaming response from an AI backend."""

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
