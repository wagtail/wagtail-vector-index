from .base import VectorIndex
from .model import (
    register_indexed_models,
)
from .registry import registry


def get_vector_indexes() -> dict[str, VectorIndex]:
    register_indexed_models()

    return {name: cls(object_type=cls) for name, cls in registry._registry.items()}
