import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, NotRequired, Self

import llm
from llm.models import dataclass

from ..types import AIResponse
from .base import (
    BaseChatBackend,
    BaseChatConfig,
    BaseChatConfigSettingsDict,
    BaseConfigSettingsDict,
    BaseEmbeddingBackend,
    BaseEmbeddingConfig,
    BaseEmbeddingConfigSettingsDict,
)


class BaseLLMSettingsDict(BaseConfigSettingsDict):
    INIT_KWARGS: NotRequired[Mapping[str, Any] | None]


class LLMBackendSettingsDict(BaseLLMSettingsDict, BaseChatConfigSettingsDict):
    PROMPT_KWARGS: NotRequired[Mapping[str, Any] | None]


class LLMEmbeddingSettingsDict(BaseLLMSettingsDict, BaseEmbeddingConfigSettingsDict):
    pass


@dataclass(kw_only=True)
class LLMBackendConfigMixin:
    init_kwargs: Mapping[str, Any]

    @classmethod
    def from_settings(cls, config: BaseLLMSettingsDict, **kwargs: Any) -> Self:
        init_kwargs = config.get("INIT_KWARGS")
        if init_kwargs is None:
            init_kwargs = {}
        kwargs.setdefault("init_kwargs", init_kwargs)

        return super().from_settings(config, **kwargs)  # type: ignore


@dataclass(kw_only=True)
class LLMChatBackendConfig(
    LLMBackendConfigMixin, BaseChatConfig[LLMBackendSettingsDict]
):
    prompt_kwargs: Mapping[str, Any]

    @classmethod
    def from_settings(cls, config: LLMEmbeddingSettingsDict, **kwargs: Any) -> Self:
        prompt_kwargs = config.get("PROMPT_KWARGS")
        if prompt_kwargs is None:
            prompt_kwargs = {}
        kwargs.setdefault("prompt_kwargs", prompt_kwargs)

        return super().from_settings(config, **kwargs)


@dataclass(kw_only=True)
class LLMEmbeddingBackendConfig(
    LLMBackendConfigMixin, BaseEmbeddingConfig[LLMEmbeddingSettingsDict]
):
    @classmethod
    def from_settings(cls, config: LLMEmbeddingSettingsDict, **kwargs: Any) -> Self:
        init_kwargs = config.get("INIT_KWARGS")
        if init_kwargs is None:
            init_kwargs = {}
        kwargs.setdefault("init_kwargs", init_kwargs)

        return super().from_settings(config, **kwargs)


class LLMChatBackend(BaseChatBackend[LLMChatBackendConfig]):
    config: LLMChatBackendConfig
    config_cls = LLMChatBackendConfig

    def chat(self, *, user_messages: Sequence[str]) -> AIResponse:
        model = self._get_llm_chat_model()
        full_prompt = os.linesep.join(user_messages)
        return model.prompt(full_prompt, **self._get_prompt_kwargs())

    def _get_prompt_kwargs(self, **prompt_kwargs: Any) -> Mapping[str, Any]:
        prompt_kwargs = {}
        if self.config.prompt_kwargs is not None:
            prompt_kwargs.update(self.config.prompt_kwargs)
        return prompt_kwargs

    def _get_llm_chat_model(self) -> llm.Model:
        model = llm.get_model(self.config.model_id)
        if self.config.init_kwargs is not None:
            for config_key, config_val in self.config.init_kwargs.items():
                setattr(model, config_key, config_val)
        return model


class LLMEmbeddingBackend(BaseEmbeddingBackend[LLMEmbeddingBackendConfig]):
    config: LLMEmbeddingBackendConfig
    config_cls = LLMEmbeddingBackendConfig

    def _get_llm_embedding_model(self) -> llm.EmbeddingModel:
        model = llm.get_embedding_model(self.config.model_id)
        if self.config.init_kwargs is not None:
            for config_key, config_val in self.config.init_kwargs.items():
                setattr(model, config_key, config_val)
        return model

    def embed(self, inputs: Iterable[str]) -> Iterator[list[float]]:
        model = self._get_llm_embedding_model()
        yield from model.embed_multi(inputs)
