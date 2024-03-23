import copy
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any, Generic, TypeVar

from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.index.base import Document, VectorIndex
from wagtail_vector_index.index.registry import registry

ConfigClass = TypeVar("ConfigClass")
IndexClass = TypeVar("IndexClass", bound="Index")


class Index:
    def __init__(self, index_name: str, **kwargs: Any) -> None:
        self.index_name = index_name

    def get_vector_index(self) -> "VectorIndex":
        return registry[self.index_name]()

    def upsert(self, *, documents: Iterable["Document"]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def delete(self, *, document_ids: Sequence[str]) -> None:
        raise NotImplementedError

    def similarity_search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> Generator[Document, None, None]:
        raise NotImplementedError


class Backend(Generic[ConfigClass, IndexClass]):
    config: ConfigClass
    config_class: type[ConfigClass]
    index_class: type[IndexClass]

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            config = dict(copy.deepcopy(config))
            self.config = self.config_class(**config)
        except TypeError as e:
            raise ImproperlyConfigured(
                f"Missing configuration settings for the vector backend: {e}"
            ) from e

    def __init_subclass__(cls, **kwargs: Any) -> None:
        try:
            cls.config_class  # noqa: B018
        except AttributeError as e:
            raise AttributeError(
                f"Vector backend {cls.__name__} must specify a `config_class` class \
                    attribute"
            ) from e
        return super().__init_subclass__(**kwargs)

    def get_index(self, index_name: str) -> IndexClass:
        raise NotImplementedError

    def create_index(self, index_name: str, *, vector_size: int) -> IndexClass:
        raise NotImplementedError

    def delete_index(self, index_name: str):
        raise NotImplementedError
