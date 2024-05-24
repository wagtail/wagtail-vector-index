import re

import pytest
from wagtail_vector_index.ai_utils.backends import (
    InvalidAIBackendError,
    get_chat_backend,
    get_embedding_backend,
)
from wagtail_vector_index.ai_utils.backends.echo import (
    EchoChatBackend,
    EchoEmbeddingBackend,
)

###############################################################################
# Chat
###############################################################################


def test_get_configured_chat_backend_instance():
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
            },
        },
        backend_id="default",
    )
    assert isinstance(backend, EchoChatBackend)


def test_get_invalid_chat_backend_class_instance():
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("some.random.not.existing.path") is not importable.'
        ),
    ):
        get_chat_backend(
            backend_dict={
                "CLASS": "some.random.not.existing.path",
                "CONFIG": {
                    "MODEL_ID": "echo",
                    "TOKEN_LIMIT": 1024,
                },
            },
            backend_id="default",
        )


def test_get_chat_backend_instance_with_custom_setting():
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 123123,
                "MAX_WORD_SLEEP_SECONDS": 150,  # type: ignore
            },
        },
        backend_id="default",
    )
    assert isinstance(backend, EchoChatBackend)
    assert backend.config.model_id == "echo"
    assert backend.config.max_word_sleep_seconds == 150
    assert backend.config.token_limit == 123123


def test_chat():
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
                "MAX_WORD_SLEEP_SECONDS": 0,  # type: ignore
            },
        },
        backend_id="default",
    )
    input_text = "".join(
        [
            "Wagtails are a group of passerine birds that form the genus Motacilla in the family Motacillidae. The forest wagtail belongs to the monotypic genus Dendronanthus which is closely related to Motacilla and sometimes included therein. The common name and genus names are derived from their characteristic tail pumping behaviour. Together with the pipits and longclaws they form the family Motacillidae.",
            "The willie wagtail (Rhipidura leucophrys) of Australia is an unrelated bird similar in coloration and shape to the Japanese wagtail. It belongs to the fantails.",
        ]
    )
    response = backend.chat(
        messages=[
            {"content": "Translate the following context to French.", "role": "user"},
            {"content": input_text, "role": "user"},
        ]
    )
    assert (
        response.choices[0]
        == f"This is an echo backend: Translate the following context to French. {input_text}"
    )


def test_streaming_chat():
    backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
                "MAX_WORD_SLEEP_SECONDS": 0,  # type: ignore
            },
        },
        backend_id="default",
    )
    input_text = "At first glance, the wagtails appear to be divided into a yellow-bellied group and a white-bellied one, or one where the upper head is black and another where it is usually grey, but may be olive, yellow, or other colours. However, these are not evolutionary lineages; change of belly colour and increase of melanin have occurred independently several times in the wagtails, and the colour patterns which actually indicate relationships are more subtle."
    response = backend.chat(
        messages=[
            {"content": "Translate the following context to French.", "role": "user"},
            {"content": input_text, "role": "user"},
        ],
        stream=True,
    )
    assert [part["content"] for part in response] == [
        "This",
        "is",
        "an",
        "echo",
        "backend:",
        "Translate",
        "the",
        "following",
        "context",
        "to",
        "French.",
        *input_text.split(),
    ]


###############################################################################
# Embeddings
###############################################################################


def test_get_configured_embedding_backend_instance():
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 123123,
                "EMBEDDING_OUTPUT_DIMENSIONS": 12,
            },
        },
        backend_id="default",
    )
    assert backend.config.model_id == "echo"
    assert backend.config.token_limit == 123123
    assert backend.config.embedding_output_dimensions == 12
    assert isinstance(backend, EchoEmbeddingBackend)


def test_get_invalid_embedding_backend_class_instance():
    with pytest.raises(
        InvalidAIBackendError,
        match=re.escape(
            'Invalid AI backend: "AI backend "default" settings: "CLASS" '
            '("some.random.not.existing.path") is not importable.'
        ),
    ):
        get_embedding_backend(
            backend_dict={
                "CLASS": "some.random.not.existing.path",
                "CONFIG": {
                    "MODEL_ID": "echo",
                    "TOKEN_LIMIT": 1024,
                    "EMBEDDING_OUTPUT_DIMENSIONS": 10,
                },
            },
            backend_id="default",
        )


def test_get_embedding_backend_instance():
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 123123,
                "EMBEDDING_OUTPUT_DIMENSIONS": 10,
            },
        },
        backend_id="default",
    )
    assert isinstance(backend, EchoEmbeddingBackend)


def test_embed():
    backend = get_embedding_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
                "EMBEDDING_OUTPUT_DIMENSIONS": 10,
            },
        },
        backend_id="default",
    )
    response = backend.embed(
        [
            "Little trotty wagtail he went in the rain,",
            "And tittering, tottering sideways he neer got straight again,",
            "He stooped to get a worm, and looked up to get a fly,",
            "And then he flew away ere his feathers they were dry.",
        ]
    )

    counter = 0
    for r in response:
        counter += 1

        # Test embedding output dimensions.
        assert len(r) == 10

        # Test that all vector's values are floats.
        assert all(isinstance(x, float) for x in r)

    # We should get as many embeddings as we have items in the list.
    assert counter == 4
