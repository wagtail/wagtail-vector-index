import copy
from collections.abc import AsyncGenerator, Generator, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.ai_utils.backends.base import BaseEmbeddingBackend

if TYPE_CHECKING:
    from wagtail_vector_index.storage.filters import DocumentFilter
    from wagtail_vector_index.storage.models import Document

StorageProviderClass = TypeVar("StorageProviderClass")
IndexMixin = TypeVar("IndexMixin")
FromObjectType = TypeVar("FromObjectType", contravariant=True)
ChunkedObjectType = TypeVar("ChunkedObjectType", covariant=False)
ToObjectType = TypeVar("ToObjectType", covariant=True)


class DocumentRetrievalVectorIndexMixinProtocol(Protocol):
    """Protocol which defines the minimum requirements for a VectorIndex to be used with a mixin that provides
    document retrieval/generation"""

    def get_embedding_backend(self) -> BaseEmbeddingBackend: ...


class StorageVectorIndexMixinProtocol(Protocol[StorageProviderClass]):
    """Protocol which defines the minimum requirements for a VectorIndex to be used with a StorageProvider mixin."""

    storage_provider: StorageProviderClass
    _filters: list["DocumentFilter"]

    def rebuild_index(self) -> None: ...

    def upsert(self) -> None: ...

    def get_documents(self) -> Iterable["Document"]: ...

    async def aget_documents(self) -> AsyncGenerator["Document", None]: ...

    def _get_storage_provider(self) -> StorageProviderClass: ...


ConfigClass = TypeVar("ConfigClass")


class StorageProvider(Generic[ConfigClass, IndexMixin]):
    """Base class for a storage provider that provides methods for interacting with a provider,
    e.g. creating and managing indexes."""

    config: ConfigClass
    config_class: type[ConfigClass]
    index_mixin: type[IndexMixin]

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            config = dict(copy.deepcopy(config))
            self.config = self.config_class(**config)
        except TypeError as e:
            raise ImproperlyConfigured(
                f"Missing configuration settings for the vector backend: {e}"
            ) from e

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if not hasattr(cls, "config_class"):
            raise AttributeError(
                f"Storage provider {cls.__name__} must specify a `config_class` class \
                    attribute"
            )
        return super().__init_subclass__(**kwargs)


class FromDocumentOperator(Protocol[ToObjectType]):
    """Protocol for a class that can convert a Document to an object"""

    def from_document(self, document: "Document") -> ToObjectType: ...

    async def afrom_document(self, document: "Document") -> ToObjectType: ...

    def bulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> Generator[ToObjectType, None, None]: ...

    async def abulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> AsyncGenerator[ToObjectType, None]: ...


class ToDocumentOperator(Protocol[FromObjectType]):
    """Protocol for a class that can convert an object to a Document"""

    def __init__(self): ...

    def to_documents(
        self,
        objects: Iterable[FromObjectType],
        *,
        embedding_backend: BaseEmbeddingBackend,
        batch_size: int = 100,
    ) -> Generator["Document", None, None]: ...

    async def ato_documents(
        self,
        objects: Iterable[FromObjectType],
        *,
        embedding_backend: BaseEmbeddingBackend,
        batch_size: int = 100,
    ) -> AsyncGenerator["Document", None]: ...
