from collections.abc import Generator
from dataclasses import dataclass
from typing import Generic, Iterable, List

from django.conf import settings

from wagtail_vector_index.ai import get_ai_backend
from wagtail_vector_index.backends import get_vector_backend

from ..base import Document, VectorIndexableType


@dataclass
class QueryResponse(Generic[VectorIndexableType]):
    """Represents a response to the VectorIndex `query` method,
    including a response string and a list of sources that were used to generate the response
    """

    response: str
    sources: Iterable[VectorIndexableType]


class VectorIndex(Generic[VectorIndexableType]):
    """Base class for a VectorIndex, representing some set of documents that can be queried"""

    def __init__(
        self,
        *,
        object_type: type[VectorIndexableType],
        ai_backend_alias="default",
        vector_backend_alias="default",
    ):
        super().__init__()

        self.ai_backend = get_ai_backend(ai_backend_alias)
        self.vector_backend = get_vector_backend(alias=vector_backend_alias)
        self.backend_index = self.vector_backend.get_index(self.__class__.__name__)
        self.object_type = object_type

    def get_documents(self) -> Iterable[Document]:
        raise NotImplementedError

    def query(self, query: str) -> QueryResponse[VectorIndexableType]:
        """Perform a natural language query against the index, returning a QueryResponse containing the natural language response, and a list of sources"""
        query_embedding = self.ai_backend.embed([query])[0]
        similar_documents = self.backend_index.similarity_search(query_embedding)
        sources = self.object_type.bulk_from_documents(similar_documents)
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
        response = self.ai_backend.chat(system_messages=[], user_messages=user_messages)
        return QueryResponse(response=response, sources=sources)

    def similar(self, object: VectorIndexableType) -> List[VectorIndexableType]:
        """Find similar objects to the given object"""
        object_documents: Generator[Document, None, None] = object.to_documents(
            ai_backend=self.ai_backend
        )
        similar_documents = []
        for document in object_documents:
            similar_documents += self.backend_index.similarity_search(document.vector)

        return list(self.object_type.bulk_from_documents(similar_documents))

    def search(self, query: str, *, limit: int = 5) -> list[VectorIndexableType]:
        """Perform a search against the index, returning only a list of matching sources"""
        query_embedding = self.ai_backend.embed([query])[0]
        similar_documents = self.backend_index.similarity_search(
            query_embedding, limit=limit
        )
        return list(self.object_type.bulk_from_documents(similar_documents))

    def rebuild_index(self) -> None:
        """Build the index from scratch"""
        self.vector_backend.delete_index(self.__class__.__name__)
        index = self.vector_backend.create_index(
            self.__class__.__name__,
            vector_size=self.ai_backend.embedding_output_dimensions,
        )
        index.upsert(documents=self.get_documents())
