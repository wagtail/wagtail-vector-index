from wagtail_vector_index.index.base import VectorIndex
from wagtail_vector_index.index.models import (
    register_indexed_models,
)
from wagtail_vector_index.index.registry import registry


def get_vector_indexes() -> dict[str, VectorIndex]:
    register_indexed_models()

    return {name: cls() for name, cls in registry._registry.items()}
