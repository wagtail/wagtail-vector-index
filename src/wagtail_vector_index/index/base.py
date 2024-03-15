import logging
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Generic

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.conf import settings
from llm.models import Response

from wagtail_vector_index.ai import get_chat_backend, get_embedding_backend
from wagtail_vector_index.backends import get_vector_backend

from ..ai_utils.backends.base import BaseChatBackend, BaseEmbeddingBackend
from ..base import Document, VectorIndexableType

logger = logging.Logger(__name__)


@dataclass
class QueryResponse(Generic[VectorIndexableType]):
    """Represents a response to the VectorIndex `query` method,
    including a response string and a list of sources that were used to generate the response
    """

    response: str
    sources: Iterable[VectorIndexableType]


@dataclass
class AsyncQueryResponse(Generic[VectorIndexableType]):
    """Represents a response to the VectorIndex `aquery` method,
    including a response object so users can call it's iterator
    and a list of sources that were used to generate the response
    """

    response: Response
    sources: Iterable[VectorIndexableType]


@database_sync_to_async
def get_metadata_from_documents_async(similar_documents):
    metadata_list = []
    for doc in similar_documents:
        metadata_list.append(doc.metadata["content"])
    return "\n".join(metadata_list)


class VectorIndex(Generic[VectorIndexableType]):
    """Base class for a VectorIndex, representing some set of documents that can be queried"""

    embedding_backend: BaseEmbeddingBackend
    chat_backend: BaseChatBackend
    object_type: type[VectorIndexableType]

    def __init__(
        self,
        *,
        chat_backend_alias="default",
        embedding_backend_alias="default",
        vector_backend_alias="default",
    ):
        super().__init__()

        self.embedding_backend = get_embedding_backend(embedding_backend_alias)
        self.chat_backend = get_chat_backend(chat_backend_alias)
        self.vector_backend = get_vector_backend(alias=vector_backend_alias)
        self.backend_index = self.vector_backend.get_index(self.__class__.__name__)

    def get_documents(self) -> Iterable[Document]:
        raise NotImplementedError

    def query(
        self, query: str, *, sources_limit: int = 5
    ) -> QueryResponse[VectorIndexableType]:
        """Perform a natural language query against the index, returning a QueryResponse containing the natural language response, and a list of sources"""
        try:
            query_embedding = next(self.embedding_backend.embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e

        similar_documents = self.backend_index.similarity_search(query_embedding)

        sources = self._deduplicate_list(
            self.object_type.bulk_from_documents(similar_documents)
        )

        merged_context = "\n".join(doc.metadata["content"] for doc in similar_documents)
        prompt = (
            getattr(settings, "WAGTAIL_VECTOR_INDEX_QUERY_PROMPT", None)
            or "You are a helpful assistant. Use the following context to answer the question. Don't mention the context in your answer."
        )
        user_messages = [
            prompt,
            merged_context,
            query,
        ]

        response = self.chat_backend.chat(user_messages=user_messages)
        return QueryResponse(response=response.text(), sources=sources)

    async def aquery(self, query: str) -> AsyncQueryResponse[VectorIndexableType]:
        """
        Async version of the query method.
        """
        if not self.chat_backend.can_stream():
            logger.warning("Chat backend does not support streaming")

        try:
            query_embedding = next(self.embedding_backend.embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e

        similar_documents = await sync_to_async(self.backend_index.similarity_search)(
            query_embedding
        )

        sources = await sync_to_async(self._deduplicate_list)(
            self.object_type.bulk_from_documents(similar_documents)
        )

        merged_context = await get_metadata_from_documents_async(similar_documents)

        prompt = (
            getattr(settings, "WAGTAIL_VECTOR_INDEX_QUERY_PROMPT", None)
            or "You are a helpful assistant. Use the following context to answer the question. Don't mention the context in your answer."
        )
        user_messages = [
            prompt,
            merged_context,
            query,
        ]

        response = self.chat_backend.chat(user_messages=user_messages)
        return AsyncQueryResponse(response=response, sources=sources)

    def similar(
        self, object: VectorIndexableType, *, include_self: bool = False, limit: int = 5
    ) -> list[VectorIndexableType]:
        """Find similar objects to the given object"""
        object_documents: Generator[Document, None, None] = object.to_documents(
            embedding_backend=self.embedding_backend
        )
        similar_documents = []
        for document in object_documents:
            similar_documents += self.backend_index.similarity_search(
                document.vector, limit=limit
            )

        return self._deduplicate_list(
            self.object_type.bulk_from_documents(similar_documents),
            exclusions=None if include_self else [object],
        )

    def search(self, query: str, *, limit: int = 5) -> list[VectorIndexableType]:
        """Perform a search against the index, returning only a list of matching sources"""
        try:
            query_embedding = next(self.embedding_backend.embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e
        similar_documents = self.backend_index.similarity_search(
            query_embedding, limit=limit
        )

        # Eliminate duplicates of the same objects.
        return self._deduplicate_list(
            self.object_type.bulk_from_documents(similar_documents)
        )

    @staticmethod
    def _deduplicate_list(
        documents: Iterable[VectorIndexableType],
        *,
        exclusions: Iterable[VectorIndexableType] | None = None,
    ) -> list[VectorIndexableType]:
        if exclusions is None:
            exclusions = []
        # This code assumes that dict.fromkeys preserves order which is
        # behavior of the Python language since version 3.7.
        return list(dict.fromkeys(item for item in documents if item not in exclusions))

    def rebuild_index(self) -> None:
        """Build the index from scratch"""
        self.vector_backend.delete_index(self.__class__.__name__)
        index = self.vector_backend.create_index(
            self.__class__.__name__,
            vector_size=self.embedding_backend.embedding_output_dimensions,
        )
        index.upsert(documents=self.get_documents())
