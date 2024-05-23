import re
from typing import List

import pytest
from wagtail_vector_index.ai_utils.backends import (
    InvalidAIBackendError,
    get_chat_backend,
    get_embedding_backend,
)
from wagtail_vector_index.ai_utils.types import ChatMessage

try:
    import litellm  # noqa: F401
except ImportError:
    litellm_installed = False
else:
    litellm_installed = True


if_litellm_installed = pytest.mark.skipif(
    not litellm_installed, reason="Requires litellm to be installed."
)
if_litellm_missing = pytest.mark.skipif(
    litellm_installed, reason="Requires litellm to be not installed."
)


@pytest.fixture
def litellm_chat_backend_class():
    from wagtail_vector_index.ai_utils.backends.litellm import LiteLLMChatBackend

    return LiteLLMChatBackend


@pytest.fixture
def litellm_embedding_backend_class():
    from wagtail_vector_index.ai_utils.backends.litellm import LiteLLMEmbeddingBackend

    return LiteLLMEmbeddingBackend


@pytest.fixture
def make_chat_backend():
    def _make_chat_backend(default_parameters: dict[str, str] | None = None):
        return get_chat_backend(
            backend_dict={
                "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMChatBackend",
                "CONFIG": {
                    "MODEL_ID": "gpt-3.5-turbo",
                    "DEFAULT_PARAMETERS": default_parameters or {},  # type: ignore
                },
            },
            backend_id="default",
        )

    return _make_chat_backend


@pytest.fixture
def make_embedding_backend():
    def _make_embedding_backend(default_parameters: dict[str, str] | None = None):
        return get_embedding_backend(
            backend_dict={
                "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMEmbeddingBackend",
                "CONFIG": {
                    "MODEL_ID": "text-embedding-ada-002",
                    "DEFAULT_PARAMETERS": default_parameters or {},  # type: ignore
                },
            },
            backend_id="default",
        )

    return _make_embedding_backend


###############################################################################
# Chat
###############################################################################


@if_litellm_missing
def test_chat_backend_import_error(make_chat_backend):
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("wagtail_vector_index.ai_utils.backends.litellm.LiteLLMChatBackend") is not importable.'
        ),
    ):
        make_chat_backend()


@if_litellm_missing
def test_get_configured_chat_backend_instance(
    make_chat_backend, litellm_chat_backend_class
):
    backend = make_chat_backend()
    assert isinstance(backend, litellm_chat_backend_class)


@if_litellm_installed
def test_litellm_custom_chat_default_parameters(make_chat_backend, mocker):
    backend = make_chat_backend(default_parameters={"api_key": "random-api-key"})
    prompt_mock = mocker.patch(
        "wagtail_vector_index.ai_utils.backends.litellm.litellm.completion"
    )
    input_text = [
        "Little trotty wagtail, he waddled in the mud,",
        "And left his little footmarks, trample where he would.",
        "He waddled in the water-pudge, and waggle went his tail,",
        "And chirrupt up his wings to dry upon the garden rail.",
    ]
    messages: List[ChatMessage] = [
        {"content": message, "role": "user"} for message in input_text
    ]
    backend.chat(messages=messages)
    prompt_mock.assert_called_once_with(
        messages=messages, api_key="random-api-key", model="gpt-3.5-turbo", stream=False
    )


@if_litellm_missing
def test_litellm_custom_chat_default_parameters_overridable(make_chat_backend, mocker):
    backend = make_chat_backend(default_parameters={"api_key": "random-api-key"})
    prompt_mock = mocker.patch(
        "wagtail_vector_index.ai_utils.backends.litellm.litellm.completion"
    )
    input_text = [
        "Little trotty wagtail, he waddled in the mud,",
        "And left his little footmarks, trample where he would.",
        "He waddled in the water-pudge, and waggle went his tail,",
        "And chirrupt up his wings to dry upon the garden rail.",
    ]
    messages: List[ChatMessage] = [
        {"content": message, "role": "user"} for message in input_text
    ]
    backend.chat(messages=messages, api_key="other-api-key")
    prompt_mock.assert_called_once_with(
        messages=messages, api_key="other-api-key", model="gpt-3.5-turbo", stream=False
    )


@if_litellm_installed
def test_litellm_chat(make_chat_backend):
    backend = make_chat_backend()
    input_text = "Little trotty wagtail, he waddled in the mud."
    response_text = "And left his little footmarks, trample where he would."
    messages: List[ChatMessage] = [
        {"content": message, "role": "user"} for message in input_text
    ]
    response = backend.chat(messages=messages, mock_response=response_text)
    assert next(response) == response_text


@if_litellm_installed
async def test_litellm_async_chat(make_chat_backend):
    backend = make_chat_backend()
    input_text = "Little trotty wagtail, he waddled in the mud."
    response_text = "And left his little footmarks, trample where he would."
    messages: List[ChatMessage] = [
        {"content": message, "role": "user"} for message in input_text
    ]
    response = await backend.achat(messages=messages, mock_response=response_text)
    assert next(response) == response_text


###############################################################################
# Embeddings
###############################################################################


@if_litellm_missing
def test_embedding_backend_import_error(make_embedding_backend):
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("wagtail_vector_index.ai_utils.backends.litellm.LiteLLMEmbeddingBackend") is not importable.'
        ),
    ):
        make_embedding_backend()


@if_litellm_installed
def test_get_configured_embedding_backend_instance(
    make_embedding_backend, litellm_embedding_backend_class
):
    backend = make_embedding_backend()
    assert isinstance(backend, litellm_embedding_backend_class)


@if_litellm_installed
def test_litellm_custom_embedding_default_parameters(make_embedding_backend):
    backend = make_embedding_backend(default_parameters={"api_key": "random-api-key"})
    assert backend.config.model_id == "text-embedding-ada-002"
    assert backend.config.token_limit == 8191
    assert backend.config.embedding_output_dimensions == 1536
    assert backend.config.default_parameters == {"api_key": "random-api-key"}


@if_litellm_installed
def test_litellm_embed(make_embedding_backend, mocker):
    backend = make_embedding_backend()
    embed_mock = mocker.patch(
        "wagtail_vector_index.ai_utils.backends.litellm.litellm.embedding"
    )
    input_text = [
        "Little trotty wagtail, he waddled in the mud,",
        "And left his little footmarks, trample where he would.",
        "He waddled in the water-pudge, and waggle went his tail,",
        "And chirrupt up his wings to dry upon the garden rail.",
    ]
    list(backend.embed(input_text))
    embed_mock.assert_called_once_with(inputs=input_text, model=backend.config.model_id)
