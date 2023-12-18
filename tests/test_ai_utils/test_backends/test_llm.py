import os
import re

import pytest
from wagtail_vector_index.ai_utils.backends import (
    InvalidAIBackendError,
    get_chat_backend,
    get_embedding_backend,
)

try:
    import llm  # noqa: F401
except ImportError:
    llm_installed = False
else:
    llm_installed = True


skip_if_llm_not_installed = pytest.mark.skipif(
    not llm_installed, reason="Requires llm to be installed."
)
skip_if_llm_installed = pytest.mark.skipif(
    llm_installed, reason="Requires llm to be not installed."
)


@pytest.fixture
def llm_chat_backend_class():
    from wagtail_vector_index.ai_utils.backends.llm import LLMChatBackend

    return LLMChatBackend


@pytest.fixture
def llm_embedding_backend_class():
    from wagtail_vector_index.ai_utils.backends.llm import LLMEmbeddingBackend

    return LLMEmbeddingBackend


###############################################################################
# Chat
###############################################################################


@skip_if_llm_installed
def test_chat_backend_import_error():
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("wagtail_vector_index.ai_utils.backends.llm.LLMChatBackend") is not importable.'
        ),
    ):
        get_chat_backend(
            backend_dict={
                "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMChatBackend",
                "CONFIG": {
                    "MODEL_ID": "gpt-3.5-turbo",
                    "INIT_KWARGS": {"key": "random-api-key"},  # type: ignore
                },
            },
            backend_id="default",
        )


@skip_if_llm_not_installed
def test_get_configured_chat_backend_instance(llm_chat_backend_class):
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-3.5-turbo",
            },
        },
        backend_id="default",
    )
    assert isinstance(backend, llm_chat_backend_class)


@skip_if_llm_not_installed
def test_llm_custom_chat_init_kwargs():
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-3.5-turbo",
                "INIT_KWARGS": {"key": "random-api-key"},  # type: ignore
            },
        },
        backend_id="default",
    )
    assert backend.config.model_id == "gpt-3.5-turbo"
    assert backend.config.token_limit == 4096
    assert backend.config.init_kwargs == {"key": "random-api-key"}
    llm_model = backend._get_llm_chat_model()  # type: ignore
    assert llm_model.key == "random-api-key"


@skip_if_llm_not_installed
def test_llm_prompt_with_custom_kwargs(mocker):
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-3.5-turbo",
                "PROMPT_KWARGS": {  # type: ignore
                    "system": "This is a test system prompt."
                },
            },
        },
        backend_id="default",
    )
    assert backend.config.prompt_kwargs == {"system": "This is a test system prompt."}
    prompt_mock = mocker.patch(
        "wagtail_vector_index.ai_utils.backends.llm.llm.models.Model.prompt"
    )
    input_text = [
        "Little trotty wagtail, he waddled in the mud,",
        "And left his little footmarks, trample where he would.",
        "He waddled in the water-pudge, and waggle went his tail,",
        "And chirrupt up his wings to dry upon the garden rail.",
    ]
    backend.chat(user_messages=input_text)
    prompt_mock.assert_called_once_with(
        os.linesep.join(input_text),
        system="This is a test system prompt.",
    )


###############################################################################
# Embeddings
###############################################################################


@skip_if_llm_installed
def test_embedding_backend_import_error():
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("wagtail_vector_index.ai_utils.backends.llm.LLMEmbeddingBackend") is not importable.'
        ),
    ):
        get_embedding_backend(
            backend_dict={
                "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMEmbeddingBackend",
                "CONFIG": {
                    "MODEL_ID": "ada-002",
                    "INIT_KWARGS": {"key": "random-api-key"},  # type: ignore
                },
            },
            backend_id="default",
        )


@skip_if_llm_not_installed
def test_get_configured_embedding_backend_instance(llm_embedding_backend_class):
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "ada-002",
                "INIT_KWARGS": {"key": "random-api-key"},  # type: ignore
            },
        },
        backend_id="default",
    )
    assert isinstance(backend, llm_embedding_backend_class)


@skip_if_llm_not_installed
def test_llm_custom_embedding_init_kwargs():
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "ada-002",
                "INIT_KWARGS": {"key": "random-api-key"},  # type: ignore
            },
        },
        backend_id="default",
    )
    assert backend.config.model_id == "ada-002"
    assert backend.config.token_limit == 8191
    assert backend.config.embedding_output_dimensions == 1536
    assert backend.config.init_kwargs == {"key": "random-api-key"}
    llm_model = backend._get_llm_embedding_model()  # type: ignore
    assert llm_model.key == "random-api-key"


@skip_if_llm_not_installed
def test_llm_embed(mocker):
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.llm.LLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "ada-002",
            },
        },
        backend_id="default",
    )
    embed_mock = mocker.patch(
        "wagtail_vector_index.ai_utils.backends.llm.llm.models.EmbeddingModel.embed_multi"
    )
    input_text = [
        "Little trotty wagtail, he waddled in the mud,",
        "And left his little footmarks, trample where he would.",
        "He waddled in the water-pudge, and waggle went his tail,",
        "And chirrupt up his wings to dry upon the garden rail.",
    ]
    # embed_multi gets called only when iterating so we are converting it to a
    # list to force the iteration.
    list(backend.embed(input_text))
    embed_mock.assert_called_once_with(input_text)
