import copy
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from wagtail_vector_index.ai import get_ai_backend

if TYPE_CHECKING:
    from wagtail_vector_index.base import Document
    from wagtail_vector_index.index.base import VectorIndex


ConfigClass = TypeVar("ConfigClass")
IndexClass = TypeVar("IndexClass", bound="Index")


class InvalidVectorBackendError(ImproperlyConfigured):
    pass


@dataclass(frozen=True, eq=True)
class SearchResponseDocument:
    id: str | int
    metadata: dict


class Index:
    def __init__(self, index_name: str, **kwargs: Any) -> None:
        self.index_name = index_name

    def get_vector_index(self) -> "VectorIndex":
        from wagtail_vector_index.index import get_vector_indexes

        # TODO: Consider passing a vector index instance to the constructor.
        return get_vector_indexes()[self.index_name]

    def upsert(self, *, documents: Iterable["Document"]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def delete(self, *, document_ids: Sequence[str]) -> None:
        raise NotImplementedError

    def similarity_search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> Generator[SearchResponseDocument, None, None]:
        raise NotImplementedError


class Backend(Generic[ConfigClass, IndexClass]):
    config: ConfigClass
    config_class: type[ConfigClass]
    index_class: type[IndexClass]

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            config = dict(copy.deepcopy(config))
            ai_backend_alias = config.pop("AI_BACKEND", "default")
            self.ai_backend = get_ai_backend(ai_backend_alias)
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


def get_vector_backend_config() -> Mapping:
    try:
        return settings.WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS
    except AttributeError:
        return {
            "default": {
                "BACKEND": "wagtail_vector_index.backends.numpy.NumpyBackend",
            }
        }


def get_vector_backend(*, alias="default") -> Backend:
    backend_config = get_vector_backend_config()

    try:
        config = backend_config[alias]
    except KeyError as e:
        raise InvalidVectorBackendError(
            f"No vector backend with alias '{alias}': {e}"
        ) from e

    try:
        imported = import_string(config["BACKEND"])
    except ImportError as e:
        raise InvalidVectorBackendError(
            f"Couldn't import backend {config['BACKEND']}: {e}"
        ) from e

    params = config.copy()
    params.pop("BACKEND")

    return imported(params)
