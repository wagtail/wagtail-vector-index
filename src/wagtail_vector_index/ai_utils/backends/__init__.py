from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, Required, TypedDict, cast

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from ..text_splitting.langchain import LangchainRecursiveCharacterTextSplitter
from ..text_splitting.naive import NaiveTextSplitterCalculator
from ..types import TextSplitterLengthCalculatorProtocol, TextSplitterProtocol
from .base import (
    BaseBackend,
    BaseChatBackend,
    BaseConfigSettingsDict,
    BaseEmbeddingBackend,
)


class TextSplittingSettingsDict(TypedDict):
    SPLITTER_CLASS: NotRequired[str]
    SPLITTER_LENGTH_CALCULATOR_CLASS: NotRequired[str]


class InvalidAIBackendError(ImproperlyConfigured):
    def __init__(self, alias):
        super().__init__(f"Invalid AI backend: {alias}")


class BackendSettingsDict(TypedDict):
    CLASS: Required[str]
    CONFIG: Required[BaseConfigSettingsDict]
    TEXT_SPLITTING: NotRequired[TextSplittingSettingsDict]


class ChatBackendSettingsDict(BackendSettingsDict):
    pass


class EmbeddingBackendSettingsDict(BackendSettingsDict):
    pass


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


@dataclass(kw_only=True)
class _TextSplitterConfig:
    splitter_class: type[TextSplitterProtocol]
    splitter_length_calculator_class: type[TextSplitterLengthCalculatorProtocol]


def _get_text_splitter_config(
    *, backend_alias: str, config: TextSplittingSettingsDict
) -> _TextSplitterConfig:
    splitter_class_path = config.get("SPLITTER_CLASS")
    length_calculator_class_path = config.get("SPLITTER_LENGTH_CALCULATOR_CLASS")

    # Text splitter - class that splits text into chunks of a given size.
    if splitter_class_path is not None:
        try:
            splitter_class = cast(
                type[TextSplitterProtocol], import_string(splitter_class_path)
            )
        except ImportError as e:
            raise ImproperlyConfigured(
                f'Cannot import "SPLITTER_CLASS" ("{splitter_class_path}") for backend "{backend_alias}".'
            ) from e
    else:
        splitter_class = _get_default_text_splitter_class()

    # Text splitter length calculator - class that calculates the number of token in a given text.
    if length_calculator_class_path is not None:
        try:
            length_calculator_class = cast(
                type[TextSplitterLengthCalculatorProtocol],
                import_string(length_calculator_class_path),
            )
        except ImportError as e:
            raise ImproperlyConfigured(
                f'Cannot import "SPLITTER_LENGTH_CALCULATOR_CLASS" ("{length_calculator_class_path}") for backend "{backend_alias}".'
            ) from e
    else:
        length_calculator_class = _get_default_text_splitter_length_class()

    return _TextSplitterConfig(
        splitter_class=splitter_class,
        splitter_length_calculator_class=length_calculator_class,
    )


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
    text_splitting = _get_text_splitter_config(
        backend_alias=backend_id,
        config=backend_dict.get("TEXT_SPLITTING", {}),
    )
    config = backend_cls.config_cls.from_settings(
        backend_settings,
        text_splitter_class=text_splitting.splitter_class,
        text_splitter_length_calculator_class=text_splitting.splitter_length_calculator_class,
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
        _get_backend(backend_dict=backend_dict, backend_id=backend_id),
    )
