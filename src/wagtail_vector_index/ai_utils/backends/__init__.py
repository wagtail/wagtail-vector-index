from collections.abc import Mapping
from typing import Required, TypedDict, cast

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from ..text_splitting.langchain import LangchainRecursiveCharacterTextSplitter
from ..text_splitting.naive import NaiveTextSplitterCalculator
from .base import (
    BaseBackend,
    BaseChatBackend,
    BaseConfigSettingsDict,
    BaseEmbeddingBackend,
    BaseEmbeddingConfigSettingsDict,
)


class InvalidAIBackendError(ImproperlyConfigured):
    def __init__(self, alias):
        super().__init__(f"Invalid AI backend: {alias}")


class BackendSettingsDict(TypedDict):
    CLASS: Required[str]
    CONFIG: Required[BaseConfigSettingsDict]


class ChatBackendSettingsDict(BackendSettingsDict):
    pass


class EmbeddingBackendSettingsDict(BackendSettingsDict):
    CONFIG: Required[BaseEmbeddingConfigSettingsDict]


def _validate_backend_settings(
    *, settings: BackendSettingsDict, backend_id: str
) -> None:
    if "CONFIG" not in settings:
        raise ImproperlyConfigured(
            f'AI backend settings for "{backend_id}": Missing "CONFIG".'
        )
    if not isinstance(settings["CONFIG"], Mapping):
        raise ImproperlyConfigured(
            f'AI backend settings for "{backend_id}": "CONFIG" is not a Mapping.'
        )
    if "MODEL_ID" not in settings["CONFIG"]:
        raise ImproperlyConfigured(
            f'AI backend settings for "{backend_id}": "MODEL_ID" is missing in "CONFIG".'
        )


def _get_default_text_splitter_class() -> type[LangchainRecursiveCharacterTextSplitter]:
    return LangchainRecursiveCharacterTextSplitter


def _get_default_text_splitter_length_class() -> type[NaiveTextSplitterCalculator]:
    return NaiveTextSplitterCalculator


def _get_backend(*, backend_dict: BackendSettingsDict, backend_id: str) -> BaseBackend:
    if "CLASS" not in backend_dict:
        raise ImproperlyConfigured(
            f'"AI backend "{backend_id}" settings: "CLASS" is missing in the configuration.'
        )

    try:
        backend_cls = cast(type[BaseBackend], import_string(backend_dict["CLASS"]))
    except ImportError as e:
        raise InvalidAIBackendError(
            f'"AI backend "{backend_id}" settings: "CLASS" ("{backend_dict["CLASS"]}") is not importable.'
        ) from e

    _validate_backend_settings(settings=backend_dict, backend_id=backend_id)

    backend_settings = backend_dict["CONFIG"]
    config = backend_cls.config_cls.from_settings(
        backend_settings,
    )

    return backend_cls(config=config)


def get_chat_backend(
    *, backend_dict: ChatBackendSettingsDict, backend_id: str
) -> BaseChatBackend:
    return cast(
        BaseChatBackend, _get_backend(backend_dict=backend_dict, backend_id=backend_id)
    )


def get_embedding_backend(
    *, backend_dict: EmbeddingBackendSettingsDict, backend_id: str
) -> BaseEmbeddingBackend:
    return cast(
        BaseEmbeddingBackend,
        _get_backend(
            backend_dict=backend_dict,  # type: ignore
            backend_id=backend_id,
        ),
    )
