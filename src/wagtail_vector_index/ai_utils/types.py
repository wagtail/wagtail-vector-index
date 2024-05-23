from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypedDict


class ChatMessage(TypedDict):
    """Dict contaning 'role' and 'content' keys representing a single message in
    a chat conversation."""

    role: str
    content: str


class AIResponse(Protocol):
    """Iterator representing a response from an AI backend"""

    def __next__(self) -> str: ...

    async def __anext__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...


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
