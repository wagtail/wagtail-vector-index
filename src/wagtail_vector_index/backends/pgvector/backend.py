import enum
import logging
from collections.abc import Generator, Iterable, MutableSequence, Sequence
from dataclasses import dataclass

from django.db.models import Func as DatabaseFunc
from django.db.models import QuerySet
from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct

from wagtail_vector_index.backends import Backend, Index, SearchResponseDocument
from wagtail_vector_index.base import Document

from .models import PgvectorEmbedding

logger = logging.Logger(__name__)

__all__ = [
    "PgvectorIndex",
    "PgvectorBackend",
    "PgvectorBackendConfig",
]


@dataclass
class PgvectorBackendConfig:
    ...


class PgvectorIndex(Index):
    class DistanceMethod(enum.Enum):
        EUCLIDEAN = "euclidean"
        COSINE = "cosine"
        MAX_INNER_PRODUCT = "max_inner_product"

    upsert_batch_size: int = 500
    bulk_create_batch_size: int = 500
    bulk_create_ignore_conflicts: bool = True
    distance_method: DistanceMethod = DistanceMethod.COSINE

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
    ) -> Generator[SearchResponseDocument, None, None]:
        for pgvector_embedding in (
            self._get_queryset()
            .select_related("embedding")
            .annotate(distance=self._distance_method_cls()("vector", query_vector))
            .order_by("distance")[:limit]
            .iterator()
        ):
            embedding = pgvector_embedding.embedding
            doc = embedding.to_document()
            yield SearchResponseDocument(id=doc.id, metadata=doc.metadata)

    def _get_queryset(self) -> QuerySet[PgvectorEmbedding]:
        return PgvectorEmbedding.objects.index(self.index_name)

    def _bulk_create(self, embeddings: Sequence[PgvectorEmbedding]) -> None:
        PgvectorEmbedding.objects.bulk_create(
            embeddings,
            batch_size=self.bulk_create_batch_size,
            ignore_conflicts=self.bulk_create_ignore_conflicts,
        )

    def _document_to_embedding(self, document: Document) -> PgvectorEmbedding:
        return PgvectorEmbedding(
            embedding_id=document.id,
            vector=document.vector,
            index_name=self.index_name,
        )

    def _distance_method_cls(self) -> type[DatabaseFunc]:
        match self.distance_method:
            case self.DistanceMethod.EUCLIDEAN:
                return L2Distance
            case self.DistanceMethod.COSINE:
                return CosineDistance
            case self.DistanceMethod.MAX_INNER_PRODUCT:
                return MaxInnerProduct
        raise ValueError(f"Invalid distance method configured: {self.distance_method}")


class PgvectorBackend(Backend[PgvectorBackendConfig, PgvectorIndex]):
    config_class = PgvectorBackendConfig

    def get_index(self, index_name: str) -> PgvectorIndex:
        return PgvectorIndex(index_name)

    def create_index(self, index_name: str, *, vector_size: int) -> PgvectorIndex:
        return self.get_index(index_name)

    def delete_index(self, index_name: str) -> None:
        self.get_index(index_name).clear()
