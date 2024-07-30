import logging
from collections.abc import (
    AsyncGenerator,
    Generator,
    Iterable,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from wagtail_vector_index.storage.base import (
    Document,
    StorageProvider,
    StorageVectorIndexMixinProtocol,
)

from .types import DistanceMethod

if TYPE_CHECKING:
    from .models import PgvectorEmbedding, PgvectorEmbeddingQuerySet

    MixinBase = StorageVectorIndexMixinProtocol["PgvectorStorageProvider"]
else:
    MixinBase = object

logger = logging.Logger(__name__)

__all__ = [
    "PgvectorStorageProvider",
    "PgvectorStorageProviderConfig",
]


def _embedding_model():
    """Lazy load the model to prevent Django trying to import it before the app registry is ready."""
    from .models import PgvectorEmbedding

    return PgvectorEmbedding


@dataclass(kw_only=True)
class PgvectorStorageProviderConfig:
    upsert_batch_size: int = 500
    bulk_create_batch_size: int = 500
    bulk_create_ignore_conflicts: bool = True
    distance_method: DistanceMethod = DistanceMethod.COSINE

    def __init__(self, **kwargs: Any) -> None:
        self.distance_method = DistanceMethod(kwargs.get("distance_method", "cosine"))
        super().__init__(**kwargs)


class PgvectorIndexMixin(MixinBase):
    upsert_batch_size: ClassVar[int] = 500
    bulk_create_batch_size: ClassVar[int] = 500
    bulk_create_ignore_conflicts: ClassVar[bool] = True
    distance_method: ClassVar[DistanceMethod] = DistanceMethod.COSINE

    def rebuild_index(self) -> None:
        self.clear()
        self.upsert(documents=self.get_documents())

    def upsert(self, *, documents: Iterable[Document]) -> None:
        counter = 0
        objs_to_create: MutableSequence["PgvectorEmbedding"] = []
        for document in documents:
            objs_to_create.append(self._document_to_embedding(document))
            if (counter + 1) % self.upsert_batch_size == 0:
                self._bulk_create(objs_to_create)
                objs_to_create = []
            counter += 1
        if objs_to_create:
            self._bulk_create(objs_to_create)

    def delete(self, *, document_ids: Sequence[str]) -> None:
        self._get_queryset().filter(embedding__pk__in=document_ids).delete()

    def clear(self):
        self._get_queryset().delete()

    def get_similar_documents(
        self, query_vector, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> Generator[Document, None, None]:
        for pgvector_embedding in self._get_similar_documents_queryset(
            query_vector, limit=limit, similarity_threshold=similarity_threshold
        ).iterator():
            embedding = pgvector_embedding.embedding
            yield embedding.to_document()

    async def aget_similar_documents(
        self, query_vector, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator[Document, None]:
        async for pgvector_embedding in self._get_similar_documents_queryset(
            query_vector, limit=limit, similarity_threshold=similarity_threshold
        ):
            embedding = pgvector_embedding.embedding
            yield embedding.to_document()

    def _get_queryset(self) -> "PgvectorEmbeddingQuerySet":
        # objects is technically a Manager instance but we want to use the custom
        # queryset method
        return cast("PgvectorEmbeddingQuerySet", _embedding_model().objects).in_index(
            type(self).__name__
        )

    def _get_similar_documents_queryset(
        self, query_vector: Sequence[float], *, limit: int, similarity_threshold: float
    ) -> "PgvectorEmbeddingQuerySet":
        queryset = (
            self._get_queryset()
            .select_related("embedding")
            .filter(embedding_output_dimensions=len(query_vector))
            .order_by_distance(
                query_vector,
                distance_method=self.distance_method,
                fetch_distance=True,
            )
        )
        if similarity_threshold > 0.0:
            # Convert similarity threshold to distance threshold
            distance_threshold = 1 - similarity_threshold
            queryset = queryset.filter(distance__lte=distance_threshold)
        return queryset[:limit]

    def _bulk_create(self, embeddings: Sequence["PgvectorEmbedding"]) -> None:
        _embedding_model().objects.bulk_create(
            embeddings,
            batch_size=self.bulk_create_batch_size,
            ignore_conflicts=self.bulk_create_ignore_conflicts,
        )

    def _document_to_embedding(self, document: Document) -> "PgvectorEmbedding":
        return _embedding_model()(
            embedding_id=document.embedding_pk,
            embedding_output_dimensions=len(document.vector),
            vector=document.vector,
            index_name=type(self).__name__,
        )


class PgvectorStorageProvider(
    StorageProvider[PgvectorStorageProviderConfig, PgvectorIndexMixin]
):
    config_class = PgvectorStorageProviderConfig
    index_mixin = PgvectorIndexMixin
