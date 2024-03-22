from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from django.conf import settings

from wagtail_vector_index.ai import get_chat_backend, get_embedding_backend
from wagtail_vector_index.ai_utils.backends.base import BaseEmbeddingBackend
from wagtail_vector_index.backends import get_vector_backend


@dataclass(kw_only=True, frozen=True)
class Document:
    """Representation of some content that is passed to vector storage backends.

    A document is usually a part of a model, e.g. some content split out from
    a VectorIndexedMixin model. One model instance may have multiple documents.

    The embedding_pk on a Document must be the PK of an Embedding model instance.
    """

    vector: Sequence[float]
    embedding_pk: int
    metadata: Mapping


class DocumentConverter(Protocol):
    def to_documents(
        self, object: object, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]: ...

    def from_document(self, document: Document) -> object: ...

    def bulk_to_documents(
        self, objects: Iterable[object], *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]: ...

    def bulk_from_documents(
        self, documents: Iterable[Document]
    ) -> Generator[object, None, None]: ...


@dataclass
class QueryResponse:
    """Represents a response to the VectorIndex `query` method,
    including a response string and a list of sources that were used to generate the response
    """

    response: str
    sources: Iterable[object]


class VectorIndex:
    """Base class for a VectorIndex, representing some set of documents that can be queried"""

    # The alias of the backend to use for generating embeddings when documents are added to this index
    embedding_backend_alias: ClassVar[str] = "default"
    # The alias of the backend (vector database) to use for storing and querying the index
    vector_backend_alias: ClassVar[str] = "default"

    def __init__(
        self,
    ):
        super().__init__()

        self.embedding_backend = get_embedding_backend(self.embedding_backend_alias)
        self.vector_backend = get_vector_backend(self.vector_backend_alias)

        self.backend_index = self.vector_backend.get_index(self.__class__.__name__)

    def get_documents(self) -> Iterable[Document]:
        raise NotImplementedError

    def get_converter(self) -> DocumentConverter:
        raise NotImplementedError

    def query(
        self, query: str, *, sources_limit: int = 5, chat_backend_alias: str = "default"
    ) -> QueryResponse:
        """Perform a natural language query against the index, returning a QueryResponse containing the natural language response, and a list of sources"""
        try:
            query_embedding = next(self.embedding_backend.embed([query]))
        except StopIteration as e:
            raise ValueError("No embeddings were generated for the given query.") from e

        similar_documents = self.backend_index.similarity_search(query_embedding)

        sources = self._deduplicate_list(
            self.get_converter().bulk_from_documents(similar_documents)
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
        chat_backend = get_chat_backend(chat_backend_alias)
        response = chat_backend.chat(user_messages=user_messages)
        return QueryResponse(response=response.text(), sources=sources)

    def similar(self, object, *, include_self: bool = False, limit: int = 5) -> list:
        """Find similar objects to the given object"""
        converter = self.get_converter()
        object_documents: Generator[Document, None, None] = converter.to_documents(
            object, embedding_backend=self.embedding_backend
        )
        similar_documents = []
        for document in object_documents:
            similar_documents += self.backend_index.similarity_search(
                document.vector, limit=limit
            )

        return self._deduplicate_list(
            converter.bulk_from_documents(similar_documents),
            exclusions=None if include_self else [object],
        )

    def search(self, query: str, *, limit: int = 5) -> list:
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
            self.get_converter().bulk_from_documents(similar_documents)
        )

    @staticmethod
    def _deduplicate_list(
        objects: Iterable[object],
        *,
        exclusions: Iterable[object] | None = None,
    ) -> list[object]:
        if exclusions is None:
            exclusions = []
        # This code assumes that dict.fromkeys preserves order which is
        # behavior of the Python language since version 3.7.
        return list(dict.fromkeys(item for item in objects if item not in exclusions))

    def rebuild_index(self) -> None:
        """Build the index from scratch"""
        self.vector_backend.delete_index(self.__class__.__name__)
        index = self.vector_backend.create_index(
            self.__class__.__name__,
            vector_size=self.embedding_backend.embedding_output_dimensions,
        )
        index.upsert(documents=self.get_documents())
