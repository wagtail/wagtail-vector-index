import copy
from abc import ABC
from collections.abc import AsyncGenerator, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, Type, TypeVar

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.ai import get_chat_backend, get_embedding_backend
from wagtail_vector_index.ai_utils.backends.base import BaseEmbeddingBackend
from wagtail_vector_index.storage import (
    get_storage_provider,
)

if TYPE_CHECKING:
    from wagtail_vector_index.storage.models import Document

StorageProviderClass = TypeVar("StorageProviderClass")
ConfigClass = TypeVar("ConfigClass")
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

    def rebuild_index(self) -> None: ...

    def upsert(self) -> None: ...

    def get_documents(self) -> Iterable["Document"]: ...

    def _get_storage_provider(self) -> StorageProviderClass: ...


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

    def bulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> Generator[ToObjectType, None, None]: ...


class ObjectChunkerOperator(Protocol[ChunkedObjectType]):
    """Protocol for a class that can chunk an object into smaller chunks"""

    def chunk_object(
        self, object: ChunkedObjectType, chunk_size: int
    ) -> Iterable[ChunkedObjectType]: ...


class ToDocumentOperator(Protocol[FromObjectType]):
    """Protocol for a class that can convert an object to a Document"""

    def __init__(self, object_chunker_operator_class: Type[ObjectChunkerOperator]): ...

    def to_documents(
        self, object: FromObjectType, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator["Document", None, None]: ...

    def bulk_to_documents(
        self,
        objects: Iterable[FromObjectType],
        *,
        embedding_backend: BaseEmbeddingBackend,
    ) -> Generator["Document", None, None]: ...


class DocumentConverter(ABC):
    """Base class for a DocumentConverter that can convert objects to Documents and vice versa"""

    to_document_operator_class: Type[ToDocumentOperator]
    from_document_operator_class: Type[FromDocumentOperator]
    object_chunker_operator_class: Type[ObjectChunkerOperator]

    @property
    def to_document_operator(self) -> ToDocumentOperator:
        return self.to_document_operator_class(self.object_chunker_operator_class)

    @property
    def from_document_operator(self) -> FromDocumentOperator:
        return self.from_document_operator_class()

    def to_documents(
        self, object: object, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator["Document", None, None]:
        return self.to_document_operator.to_documents(
            object, embedding_backend=embedding_backend
        )

    def from_document(self, document: "Document") -> object:
        return self.from_document_operator.from_document(document)

    def bulk_to_documents(
        self, objects: Iterable[object], *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator["Document", None, None]:
        return self.to_document_operator.bulk_to_documents(
            objects, embedding_backend=embedding_backend
        )

    def bulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> Generator[object, None, None]:
        return self.from_document_operator.bulk_from_documents(documents)


@dataclass
class QueryResponse:
    """Represents a response to the VectorIndex `query` method,
    including a response string and a list of sources that were used to generate the response
    """

    response: str
    sources: Iterable[object]


@dataclass
class AsyncQueryResponse:
    """Same as QueryResponse class, but with the response being an async generator."""

    response: AsyncGenerator[str, None]
    sources: Iterable[object]


class VectorIndex(Generic[ConfigClass]):
    """Base class for a VectorIndex, representing some set of documents that can be queried"""

    # The alias of the backend to use for generating embeddings when documents are added to this index
    embedding_backend_alias: ClassVar[str] = "default"

    # The alias of the storage provider specified in WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS
    storage_provider_alias: ClassVar[str] = "default"

    def get_embedding_backend(self) -> BaseEmbeddingBackend:
        return get_embedding_backend(self.embedding_backend_alias)

    def get_documents(self) -> Iterable["Document"]:
        raise NotImplementedError

    def get_converter(self) -> DocumentConverter:
        raise NotImplementedError

    # Public API

    def query(
        self,
        query: str,
        *,
        sources_limit: int = 5,
        chat_backend_alias: str = "default",
        similarity_threshold: float = 0.0,
    ) -> QueryResponse:
        """Perform a natural language query against the index, returning a QueryResponse containing the natural language response, and a list of sources"""
        try:
            query_embedding = next(self.get_embedding_backend().embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e

        similar_documents = list(
            self.get_similar_documents(
                query_embedding, similarity_threshold=similarity_threshold
            )
        )

        sources = list(self.get_converter().bulk_from_documents(similar_documents))

        merged_context = "\n".join(doc.content for doc in similar_documents)
        prompt = (
            getattr(settings, "WAGTAIL_VECTOR_INDEX_QUERY_PROMPT", None)
            or "You are a helpful assistant. Use the following context to answer the question. Don't mention the context in your answer."
        )
        messages = [
            {"content": prompt, "role": "system"},
            {"content": merged_context, "role": "system"},
            {"content": query, "role": "user"},
        ]
        chat_backend = get_chat_backend(chat_backend_alias)
        response = chat_backend.chat(messages=messages)
        return QueryResponse(response=response.choices[0], sources=sources)

    async def aquery(
        self,
        query: str,
        *,
        sources_limit: int = 5,
        chat_backend_alias: str = "default",
        similarity_threshold: float = 0.0,
    ) -> AsyncQueryResponse:
        """
        Replicates the features of `VectorIndex.query()`, but in an async way.
        """
        try:
            query_embedding = next(await self.get_embedding_backend().aembed([query]))
        except IndexError as e:
            raise ValueError("No embeddings were generated for the given query.") from e

        similar_documents = [
            doc
            async for doc in self.aget_similar_documents(
                query_embedding, similarity_threshold=similarity_threshold
            )
        ]

        sources = [
            source
            async for source in self.get_converter().abulk_from_documents(
                similar_documents
            )
        ]

        merged_context = "\n".join(doc.metadata["content"] for doc in similar_documents)
        prompt = (
            getattr(settings, "WAGTAIL_VECTOR_INDEX_QUERY_PROMPT", None)
            or "You are a helpful assistant. Use the following context to answer the question. Don't mention the context in your answer."
        )
        messages = [
            {"content": prompt, "role": "system"},
            {"content": merged_context, "role": "system"},
            {"content": query, "role": "user"},
        ]
        chat_backend = get_chat_backend(chat_backend_alias)
        response = await chat_backend.achat(messages=messages, stream=True)

        async def async_stream_wrapper():
            async for chunk in response:
                yield chunk["content"]

        return AsyncQueryResponse(
            response=async_stream_wrapper(),
            sources=sources,
        )

    def find_similar(
        self,
        object,
        *,
        include_self: bool = False,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list:
        """Find similar objects to the given object"""
        converter = self.get_converter()
        object_documents: Generator[Document, None, None] = converter.to_documents(
            object, embedding_backend=self.get_embedding_backend()
        )
        similar_documents = []
        for document in object_documents:
            similar_documents += self.get_similar_documents(
                document.vector, limit=limit, similarity_threshold=similarity_threshold
            )

        return [
            obj
            for obj in converter.bulk_from_documents(similar_documents)
            if include_self or obj != object
        ]

    def search(
        self, query: str, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> list:
        """Perform a search against the index, returning only a list of matching sources"""
        try:
            query_embedding = next(self.get_embedding_backend().embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e
        similar_documents = self.get_similar_documents(
            query_embedding, limit=limit, similarity_threshold=similarity_threshold
        )
        return list(self.get_converter().bulk_from_documents(similar_documents))

    # Utilities

    def _get_storage_provider(self):
        provider = get_storage_provider(self.storage_provider_alias)
        if not issubclass(self.__class__, provider.index_mixin):
            raise TypeError(
                f"The storage provider with alias '{self.storage_provider_alias}' requires an index that uses the '{provider.index_mixin.__class__.__name__}' mixin."
            )
        return provider

    # Backend-specific methods

    def rebuild_index(self) -> None:
        raise NotImplementedError

    def upsert(self, *, documents: Iterable["Document"]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def delete(self, *, document_ids: Sequence[str]) -> None:
        raise NotImplementedError

    def get_similar_documents(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> Generator["Document", None, None]:
        raise NotImplementedError

    def aget_similar_documents(
        self, query_vector, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator["Document", None]:
        raise NotImplementedError
