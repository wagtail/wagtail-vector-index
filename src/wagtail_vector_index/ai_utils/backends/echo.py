import random
import time
from collections.abc import Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, Self

from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.index.base import Document

from .base import (
    AIResponse,
    BaseChatBackend,
    BaseChatConfig,
    BaseChatConfigSettingsDict,
    BaseEmbeddingBackend,
    BaseEmbeddingConfig,
)


class EchoResponse(AIResponse):
    _text: str | None = None
    response_iterator: Iterator[str]

    def __init__(self, response_iterator: Iterator[str]) -> None:
        self.response_iterator = response_iterator

    def __iter__(self) -> Iterator[str]:
        return self.response_iterator

    def text(self) -> str:
        if self._text is not None:
            return self._text
        self._text = " ".join(self.response_iterator)
        return self._text


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

    def chat(self, *, user_messages: Sequence[str]) -> AIResponse:
        def response_iterator() -> Generator[str, None, None]:
            response = ["This", "is", "an", "echo", "backend:"]
            for m in user_messages:
                response.extend(m.split())
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

        return EchoResponse(response_iterator())


class EchoEmbeddingBackend(BaseEmbeddingBackend[BaseEmbeddingConfig]):
    config_cls = BaseEmbeddingConfig
    config: BaseEmbeddingConfig

    def embed(self, inputs: Iterable[Document]) -> Iterator[list[float]]:
        for _ in inputs:
            yield [
                random.random() for _ in range(self.config.embedding_output_dimensions)
            ]
