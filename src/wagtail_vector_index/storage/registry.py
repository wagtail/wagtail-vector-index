from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import VectorIndex


class VectorIndexRegistry:
    """A registry to keep track of all the VectorIndex classes that have been registered."""

    def __init__(self):
        self._registry: dict[str, "VectorIndex"] = {}

    def register_index(self, index: "VectorIndex"):
        self._registry[type(index).__name__] = index

    def __getitem__(self, key: str) -> "VectorIndex":
        return self._registry[key]

    def __iter__(self):
        return iter(self._registry.items())


registry = VectorIndexRegistry()
