from collections.abc import Mapping

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .ai_utils.backends import BackendSettingsDict, EmbeddingBackendSettingsDict
from .ai_utils.backends import get_chat_backend as get_chat_backend_instance
from .ai_utils.backends import (
    get_embedding_backend as get_embedding_backend_instance,
)
from .ai_utils.backends.base import BaseChatBackend, BaseEmbeddingBackend


def get_chat_backends_settings() -> Mapping[str, BackendSettingsDict]:
    try:
        return settings.WAGTAIL_VECTOR_INDEX["CHAT_BACKENDS"]
    except (KeyError, AttributeError) as e:
        raise ImproperlyConfigured(
            '"CHAT_BACKENDS" is missing in "WAGTAIL_VECTOR_INDEX" setting.'
        ) from e
    return {
        "default": {
            "CLASS": "ai_utils.ai.llm.LLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-3.5-turbo",
            },
        }
    }


def get_embedding_backends_settings() -> Mapping[str, EmbeddingBackendSettingsDict]:
    try:
        return settings.WAGTAIL_VECTOR_INDEX["EMBEDDING_BACKENDS"]
    except (KeyError, AttributeError) as e:
        raise ImproperlyConfigured(
            '"EMBEDDING_BACKENDS" is missing in "WAGTAIL_VECTOR_INDEX" setting.'
        ) from e
    return {
        "default": {
            "CLASS": "ai_utils.ai.llm.LLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "text-embedding-ada-002",
            },
        }
    }


def get_chat_backend(alias: str) -> BaseChatBackend:
    setting = get_chat_backends_settings()[alias]
    return get_chat_backend_instance(backend_dict=setting, backend_id=alias)


def get_embedding_backend(alias: str) -> BaseEmbeddingBackend:
    setting = get_embedding_backends_settings()[alias]
    return get_embedding_backend_instance(backend_dict=setting, backend_id=alias)
