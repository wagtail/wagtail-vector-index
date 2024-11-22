from collections.abc import Sequence
from contextlib import contextmanager

import pytest
from factories import ExampleModelFactory
from testapp.models import ExamplePage
from wagtail_vector_index.ai_utils.backends.base import (
    BaseChatBackend,
    BaseChatConfig,
    BaseEmbeddingBackend,
    BaseEmbeddingConfig,
)
from wagtail_vector_index.ai_utils.types import AIResponse
from wagtail_vector_index.storage.django import ModelKey
from wagtail_vector_index.storage.models import Document


@pytest.fixture
def patch_embedding_fields():
    @contextmanager
    def _patch_embedding_fields(model, new_embedding_fields):
        old_embedding_fields = model.embedding_fields
        model.embedding_fields = new_embedding_fields
        yield
        model.embedding_fields = old_embedding_fields

    return _patch_embedding_fields


class ChatMockBackend(BaseChatBackend):
    config_cls = BaseChatConfig

    def chat(self, messages: Sequence[str]) -> AIResponse:
        return AIResponse(choices=["AI! Don't talk to me about AI!"])


class EmbeddingMockBackend(BaseEmbeddingBackend):
    config_cls = BaseEmbeddingConfig

    def embed(self, inputs):
        values = [
            i * self.embedding_output_dimensions
            for i in range(self.embedding_output_dimensions)
        ]
        for _ in inputs:
            yield values


@pytest.fixture(autouse=True)
def use_mock_ai_backend(settings):
    settings.WAGTAIL_VECTOR_INDEX = {
        "EMBEDDING_BACKENDS": {
            "default": {
                "CLASS": "conftest.EmbeddingMockBackend",
                "CONFIG": {
                    "MODEL_ID": "mock-embedding",
                    "TOKEN_LIMIT": 1024,
                    "EMBEDDING_OUTPUT_DIMENSIONS": 6,
                },
            }
        },
        "CHAT_BACKENDS": {
            "default": {
                "CLASS": "conftest.ChatMockBackend",
                "CONFIG": {
                    "MODEL_ID": "mock-chat",
                    "TOKEN_LIMIT": 1024,
                },
            }
        },
    }


@pytest.fixture
def get_vector_for_text():
    def _get_vector_for_text(text):
        if "Very similar" in text:
            return [0.9, 0.1, 0.0]
        elif "Somewhat similar" in text:
            return [0.7, 0.3, 0.0]
        elif "test" in text.lower():
            return [1.0, 0.0, 0.0]
        else:
            return [0.1, 0.1, 0.8]

    return _get_vector_for_text


@pytest.fixture
def mock_embedding_backend(get_vector_for_text):
    class MockEmbeddingBackend(BaseEmbeddingBackend):
        def __init__(self):
            self.config = type("Config", (), {"token_limit": 100})()

        def embed(self, texts):
            def embedding_generator():
                for text in texts:
                    yield get_vector_for_text(text)

            return embedding_generator()

        async def aembed(self, texts):
            return self.embed(texts)

    return MockEmbeddingBackend()


@pytest.fixture
def test_objects():
    return [
        ExampleModelFactory.create(title="Very similar to test"),
        ExampleModelFactory.create(title="Somewhat similar to test"),
        ExampleModelFactory.create(title="Not similar at all"),
    ]


@pytest.fixture
def document_generator(test_objects, get_vector_for_text):
    def gen_documents(cls, *args, **kwargs):
        for obj in test_objects:
            vector = get_vector_for_text(obj.title)
            yield Document.objects.create(
                object_keys=[ModelKey.from_instance(obj)],
                metadata={
                    "title": obj.title,
                    "object_id": str(obj.pk),
                },
                vector=vector,
            )

    return gen_documents


@pytest.fixture
def async_document_generator(test_objects, get_vector_for_text):
    async def gen_documents(cls, *args, **kwargs):
        for obj in test_objects:
            vector = get_vector_for_text(obj.title)
            doc = await Document.objects.acreate(
                object_keys=[ModelKey.from_instance(obj)],
                metadata={"title": obj.title, "object_id": str(obj.pk)},
                vector=vector,
            )
            yield doc

    return gen_documents


@pytest.fixture
def mock_vector_index(
    mocker, mock_embedding_backend, document_generator, async_document_generator
):
    vector_index = ExamplePage.vector_index

    mocker.patch.object(
        vector_index, "get_embedding_backend", return_value=mock_embedding_backend
    )

    mocker.patch(
        "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.bulk_to_documents",
        side_effect=document_generator,
    )

    mocker.patch(
        "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.abulk_to_documents",
        side_effect=async_document_generator,
    )

    return vector_index
