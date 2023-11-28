from collections.abc import Generator
from dataclasses import dataclass
from typing import Iterable, List, Protocol, TypeVar

from .ai_utils.backends.base import BaseEmbeddingBackend

VectorIndexableType = TypeVar("VectorIndexableType", bound="VectorIndexable")


@dataclass
class Document:
    """Representation of some content that is passed to vector storage backends.

    A document is usually a part of a model, e.g. some content split out from
    a VectorIndexedMixin model. One model instance may have multiple documents.
    """

    id: str
    vector: List[float]
    metadata: dict


class VectorIndexable(Protocol[VectorIndexableType]):
    """Protocol for objects that can be converted to and from Documents, meaning they can be stored
    in a vector index."""

    def to_documents(
        self, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]:
        """Convert this object to a list of documents that can be passed to a vector storage backend"""
        ...

    @classmethod
    def from_document(cls, document: Document) -> VectorIndexableType:
        """Convert a document back to the object it represents"""
        ...

    @classmethod
    def bulk_to_documents(
        cls,
        objects: Iterable[VectorIndexableType],
        *,
        embedding_backend: BaseEmbeddingBackend,
    ) -> Iterable[Document]:
        """Convert a list of objects to a list of documents that can be passed to a vector storage backend"""
        ...

    @classmethod
    def bulk_from_documents(
        cls, documents: Iterable[Document]
    ) -> Iterable[VectorIndexableType]:
        """Convert a list of documents back to the objects they represent"""
        ...

    def __eq__(self, other: object) -> bool:
        """
        Equality check has to be implemented for the vector index to work
        properly.
        """
        ...
