from contextlib import contextmanager

import pytest

from every_ai import registry, AIBackend


@registry.register("mock")
class MockAIBackend(AIBackend):
    def __init__(self, config):
        pass

    def chat(self, prompt: str) -> str:
        return "AI! Don't talk to me about AI!"

    def embed(self, inputs):
        return [[0.1, 0.2, 0.3] for _ in range(len(inputs))]

    @property
    def embedding_output_dimensions(self):
        return 3


@pytest.fixture
def patch_embedding_fields():
    @contextmanager
    def _patch_embedding_fields(model, new_embedding_fields):
        old_embedding_fields = model.embedding_fields
        model.embedding_fields = new_embedding_fields
        yield
        model.embedding_fields = old_embedding_fields

    return _patch_embedding_fields


@pytest.fixture(autouse=True)
def use_mock_ai_backend(settings):
    settings.WAGTAIL_VECTOR_INDEX_AI_BACKENDS = {
        "default": {
            "BACKEND": "mock",
        }
    }
