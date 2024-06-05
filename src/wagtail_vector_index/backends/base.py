import copy
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.index.base import VectorIndex

ConfigClass = TypeVar("ConfigClass")
IndexClass = TypeVar("IndexClass", bound="VectorIndex")


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
