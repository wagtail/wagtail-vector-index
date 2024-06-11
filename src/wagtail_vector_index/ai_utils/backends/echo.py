"""This module contains 'Echo' backends for chat and embedding, which echo back the input.
They are intended for testing and development purposes and are not suitable for production use."
"""

import random
import time
from collections.abc import Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, Self

from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.index.base import Document

from ..types import (
    AIResponse,
    AIResponseStreamingPart,
    AIStreamingResponse,
    ChatMessage,
)
from .base import (
    BaseChatBackend,
    BaseChatConfig,
    BaseChatConfigSettingsDict,
    BaseEmbeddingBackend,
    BaseEmbeddingConfig,
)


class EchoStreamingResponse(AIResponse):
    def __init__(self, response_iterator: Iterator[str]) -> None:
        self.response_iterator = response_iterator

    def __iter__(self) -> Iterator[AIResponseStreamingPart]:
        return self

    def __next__(self) -> AIResponseStreamingPart:
        return {"index": 0, "content": next(self.response_iterator)}


@dataclass(kw_only=True)
class EchoChatSettingsDict(BaseChatConfigSettingsDict):
    MAX_WORD_SLEEP_SECONDS: NotRequired[int]


@dataclass(kw_only=True)
class EchoChatConfig(BaseChatConfig[EchoChatSettingsDict]):
    max_word_sleep_seconds: int

    @classmethod
    def from_settings(cls, config: EchoChatSettingsDict, **kwargs: Any) -> Self:
        max_word_sleep_seconds = config.get("MAX_WORD_SLEEP_SECONDS")
        if max_word_sleep_seconds is None:
            max_word_sleep_seconds = 0
        try:
            max_word_sleep_seconds = int(max_word_sleep_seconds)
        except ValueError as e:
            raise ImproperlyConfigured(
                f'"MAX_WORD_SLEEP_SECONDS" is not an "int", it is a "{type(max_word_sleep_seconds)}".'
            ) from e
        kwargs.setdefault("max_word_sleep_seconds", max_word_sleep_seconds)

        return super().from_settings(config, **kwargs)


class EchoChatBackend(BaseChatBackend[EchoChatConfig]):
    config_cls = EchoChatConfig
    config: EchoChatConfig

    def build_response(self, messages: Sequence[ChatMessage]) -> Sequence[str]:
        response = ["This", "is", "an", "echo", "backend:"]
        for m in messages:
            response.extend(m["content"].split())
        return response

    def streaming_iterator(self, response: Sequence[str]) -> Generator[str, None, None]:
        for word in response:
            if (
                self.config.max_word_sleep_seconds is not None
                and self.config.max_word_sleep_seconds > 0
            ):
                time.sleep(
                    random.random()
                    * random.randint(0, self.config.max_word_sleep_seconds)
                )
            yield word

    def chat(
        self, *, messages: Sequence[ChatMessage], stream: bool = False, **kwargs
    ) -> AIResponse | AIStreamingResponse:
        response = self.build_response(messages)
        if stream:
            return EchoStreamingResponse(self.streaming_iterator(response))
        return AIResponse(choices=[" ".join(response)])

    async def achat(
        self, *, messages: Sequence[ChatMessage], stream: bool = False, **kwargs
    ) -> AIResponse | AIStreamingResponse:
        return self.chat(messages=messages, stream=stream, **kwargs)


class EchoEmbeddingBackend(BaseEmbeddingBackend[BaseEmbeddingConfig]):
    config_cls = BaseEmbeddingConfig
    config: BaseEmbeddingConfig

    def embed(self, inputs: Iterable[Document]) -> Iterator[list[float]]:
        for _ in inputs:
            yield [
                random.random() for _ in range(self.config.embedding_output_dimensions)
            ]

    async def aembed(self, inputs: Iterable[Document]) -> Iterator[list[float]]:
        return self.embed(inputs=inputs)
