import copy
from collections.abc import AsyncGenerator, Coroutine, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Type, TypeVar, cast

from django.conf import settings
from django.db.models import QuerySet

from wagtail_vector_index.ai import get_chat_backend, get_embedding_backend
from wagtail_vector_index.ai_utils.backends.base import BaseEmbeddingBackend
from wagtail_vector_index.storage import (
    get_storage_provider,
)
from wagtail_vector_index.storage.conversion import (
    FromDocumentOperator,
    ToDocumentOperator,
)

if TYPE_CHECKING:
    from wagtail_vector_index.storage.filters import DocumentFilter
    from wagtail_vector_index.storage.models import Document


ConfigClass = TypeVar("ConfigClass")


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

    # Filters applied to the get_similar_documents method. Added to the index by the filter method.
    _filters: list["DocumentFilter"] = []

    def get_embedding_backend(self) -> BaseEmbeddingBackend:
        return get_embedding_backend(self.embedding_backend_alias)

    def get_documents(self) -> Iterable["Document"]:
        raise NotImplementedError

    async def aget_documents(self) -> AsyncGenerator["Document", None]:
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
        query_embedding = self._embed_query(query)

        similar_documents = list(
            self.get_similar_documents(
                query_embedding, similarity_threshold=similarity_threshold
            )
        )

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
        return QueryResponse(response=response.choices[0], sources=similar_documents)

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
        query_embedding = self._embed_query(query)

        similar_documents = [
            doc
            async for doc in self.aget_similar_documents(
                query_embedding, similarity_threshold=similarity_threshold
            )
        ]

        merged_context = "\n".join([doc.content for doc in similar_documents])
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
            sources=similar_documents,
        )

    def find_similar(
        self,
        object: "Document",
        *,
        include_self: bool = False,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> Iterable["Document"]:
        """Find similar documents to the given document"""
        return self.get_similar_documents(
            object.vector,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    def afind_similar(
        self,
        object: "Document",
        *,
        include_self: bool = False,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> AsyncGenerator["Document", None]:
        """Find similar objects to the given object asynchronously"""
        return self.aget_similar_documents(
            object.vector,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    def search(
        self, query: str, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> list:
        """Perform a search against the index, returning only a list of matching sources"""
        query_embedding = self._embed_query(query)
        return list(
            self.get_similar_documents(
                query_embedding, limit=limit, similarity_threshold=similarity_threshold
            )
        )

    def asearch(
        self, query: str, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator["Document", None]:
        """Perform a search against the index, returning only a list of matching sources"""
        query_embedding = self._embed_query(query)
        return self.aget_similar_documents(
            query_embedding, limit=limit, similarity_threshold=similarity_threshold
        )

    def filter(self, *filters: "DocumentFilter") -> "VectorIndex":
        """
        Returns a new VectorIndex with the given filters applied.
        """
        filtered_index = copy.copy(self)
        filtered_index._filters = list(filters)
        return filtered_index

    # Utilities
    def _embed_query(self, query: str) -> Sequence[float]:
        try:
            query_embedding = next(self.get_embedding_backend().embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e
        return query_embedding

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
    ) -> QuerySet["Document"]:
        raise NotImplementedError

    def aget_similar_documents(
        self, query_vector, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator["Document", None]:
        raise NotImplementedError


ConvertedType = TypeVar("ConvertedType")


class ConvertedVectorIndex(VectorIndex, Generic[ConvertedType]):
    """A VectorIndex which transforms documents to and from objects using the provided ToDocumentOperator and FromDocumentOperator classes"""

    to_document_operator_class: Type[ToDocumentOperator]
    from_document_operator_class: Type[FromDocumentOperator]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_document_operator = self.to_document_operator_class()
        self.from_document_operator = self.from_document_operator_class()

    # Document conversion methods
    def to_documents(
        self,
        objects: Iterable[ConvertedType],
        *,
        batch_size: int = 100,
    ) -> Generator["Document", None, None]:
        embedding_backend = self.get_embedding_backend()
        return self.to_document_operator.to_documents(
            objects, embedding_backend=embedding_backend, batch_size=batch_size
        )

    def ato_documents(
        self,
        objects: Iterable[ConvertedType],
        *,
        batch_size: int = 100,
    ) -> AsyncGenerator["Document", None]:
        embedding_backend = self.get_embedding_backend()
        return self.to_document_operator.ato_documents(
            objects, embedding_backend=embedding_backend, batch_size=batch_size
        )

    def from_document(self, document: "Document") -> ConvertedType:
        return self.from_document_operator.from_document(document)

    def afrom_document(
        self, document: "Document"
    ) -> Coroutine[ConvertedType, None, None]:
        return self.from_document_operator.afrom_document(document)

    def bulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> Generator[ConvertedType, None, None]:
        return self.from_document_operator.bulk_from_documents(documents)

    def abulk_from_documents(
        self, documents: Iterable["Document"]
    ) -> AsyncGenerator[ConvertedType, None]:
        return self.from_document_operator.abulk_from_documents(documents)

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
        response = super().query(
            query,
            sources_limit=sources_limit,
            chat_backend_alias=chat_backend_alias,
            similarity_threshold=similarity_threshold,
        )
        documents = cast(Sequence["Document"], response.sources)
        converted_sources = list(self.bulk_from_documents(documents))
        return QueryResponse(response=response.response, sources=converted_sources)

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
        response = await super().aquery(
            query,
            sources_limit=sources_limit,
            chat_backend_alias=chat_backend_alias,
            similarity_threshold=similarity_threshold,
        )

        documents = cast(Sequence["Document"], response.sources)
        converted_sources = list(self.bulk_from_documents(documents))
        return AsyncQueryResponse(response=response.response, sources=converted_sources)

    def find_similar(
        self,
        object: ConvertedType,
        *,
        include_self: bool = False,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[ConvertedType]:
        """Find similar objects to the given object"""
        document = next(self.to_documents([object]))
        similar_documents = super().find_similar(
            document,
            include_self=include_self,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        return [
            obj
            for obj in self.bulk_from_documents(similar_documents)
            if include_self or obj != object
        ]

    async def afind_similar(
        self,
        object: ConvertedType,
        *,
        include_self: bool = False,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> AsyncGenerator[ConvertedType, None]:
        """Find similar objects to the given object asynchronously"""
        document = await self.ato_documents([object]).__anext__()

        similar_documents = super().afind_similar(
            document,
            include_self=include_self,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
        similar_documents = [doc async for doc in similar_documents]

        async for obj in self.abulk_from_documents(similar_documents):
            if include_self or obj != object:
                yield obj

    def search(
        self, query: str, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> list[ConvertedType]:
        """Perform a search against the index, returning only a list of matching sources"""
        search_results = super().search(
            query, limit=limit, similarity_threshold=similarity_threshold
        )
        return [self.from_document(doc) for doc in search_results]

    async def asearch(
        self, query: str, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator[ConvertedType, None]:
        """Perform a search against the index, returning only a list of matching sources"""
        search_results = [
            doc
            async for doc in super().asearch(
                query, limit=limit, similarity_threshold=similarity_threshold
            )
        ]
        async for obj in self.abulk_from_documents(search_results):
            if obj != object:
                yield obj
