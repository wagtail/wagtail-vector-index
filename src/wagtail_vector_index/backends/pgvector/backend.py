import logging
from collections.abc import Generator, Iterable, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, cast

from wagtail_vector_index.backends.base import Backend, Index
from wagtail_vector_index.index.base import Document

from .models import PgvectorEmbedding, PgvectorEmbeddingQuerySet
from .types import DistanceMethod

logger = logging.Logger(__name__)

__all__ = [
    "PgvectorIndex",
    "PgvectorBackend",
    "PgvectorBackendConfig",
]


@dataclass(kw_only=True)
class PgvectorBackendConfig:
    upsert_batch_size: int = 500
    bulk_create_batch_size: int = 500
    bulk_create_ignore_conflicts: bool = True
    distance_method: DistanceMethod = DistanceMethod.COSINE


class PgvectorIndex(Index):
    upsert_batch_size: int
    bulk_create_batch_size: int
    bulk_create_ignore_conflicts: bool
    distance_method: DistanceMethod

    def __init__(
        self,
        index_name: str,
        upsert_batch_size: int,
        bulk_create_batch_size: int,
        bulk_create_ignore_conflicts: bool,
        distance_method: DistanceMethod | str,
        **kwargs: Any,
    ) -> None:
        super().__init__(index_name, **kwargs)
        self.upsert_batch_size = upsert_batch_size
        self.bulk_create_batch_size = bulk_create_batch_size
        self.bulk_create_ignore_conflicts = bulk_create_ignore_conflicts
        self.distance_method = DistanceMethod(distance_method)

    def upsert(self, *, documents: Iterable[Document]) -> None:
        counter = 0
        objs_to_create: MutableSequence[PgvectorEmbedding] = []
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

    def similarity_search(
        self, query_vector, *, limit: int = 5
    ) -> Generator[Document, None, None]:
        for pgvector_embedding in (
            self._get_queryset()
            .select_related("embedding")
            .filter(embedding_output_dimensions=len(query_vector))
            .order_by_distance(
                query_vector, distance_method=self.distance_method, fetch_distance=False
            )[:limit]
            .iterator()
        ):
            embedding = pgvector_embedding.embedding
            yield embedding.to_document()

    def _get_queryset(self) -> PgvectorEmbeddingQuerySet:
        # objects is technically a Manager instance but we want to use the custom
        return cast(PgvectorEmbeddingQuerySet, PgvectorEmbedding.objects).in_index(
            self.index_name
        )

    def _bulk_create(self, embeddings: Sequence[PgvectorEmbedding]) -> None:
        PgvectorEmbedding.objects.bulk_create(
            embeddings,
            batch_size=self.bulk_create_batch_size,
            ignore_conflicts=self.bulk_create_ignore_conflicts,
        )

    def _document_to_embedding(self, document: Document) -> PgvectorEmbedding:
        return PgvectorEmbedding(
            embedding_id=document.embedding_pk,
            embedding_output_dimensions=len(document.vector),
            vector=document.vector,
            index_name=self.index_name,
        )


class PgvectorBackend(Backend[PgvectorBackendConfig, PgvectorIndex]):
    config_class = PgvectorBackendConfig

    def get_index(self, index_name: str) -> PgvectorIndex:
        return PgvectorIndex(
            index_name,
            upsert_batch_size=self.config.upsert_batch_size,
            bulk_create_batch_size=self.config.bulk_create_batch_size,
            bulk_create_ignore_conflicts=self.config.bulk_create_ignore_conflicts,
            distance_method=self.config.distance_method,
        )

    def create_index(self, index_name: str, *, vector_size: int) -> PgvectorIndex:
        return self.get_index(index_name)

    def delete_index(self, index_name: str) -> None:
        self.get_index(index_name).clear()
