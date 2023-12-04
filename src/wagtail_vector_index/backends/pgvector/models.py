from collections.abc import Sequence
from typing import Self

from django.db import models
from django.utils.translation import gettext as _
from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct, VectorField

from .types import DistanceMethod


class PgvectorEmbeddingQuerySet(models.QuerySet["PgvectorEmbedding"]):
    def in_index(self, index_name: str) -> Self:
        return self.filter(index_name=index_name)

    def _distance_method_cls(
        self, distance_method: DistanceMethod | str
    ) -> type[models.Func]:
        distance_method = DistanceMethod(distance_method)

        if distance_method == DistanceMethod.COSINE:
            return CosineDistance
        elif distance_method == DistanceMethod.EUCLIDEAN:
            return L2Distance
        elif distance_method == DistanceMethod.MAX_INNER_PRODUCT:
            return MaxInnerProduct
        else:
            raise ValueError(f"Unknown distance method: {distance_method}")

    def annotate_with_distance(
        self,
        query_vector: Sequence[float],
        *,
        distance_method: DistanceMethod | str,
        fetch_distance: bool,
    ) -> Self:
        kwargs = {
            "distance": self._distance_method_cls(distance_method)(
                "vector", query_vector
            )
        }
        if fetch_distance:
            return self.annotate(**kwargs)
        return self.alias(**kwargs)

    def order_by_distance(
        self,
        query_vector: Sequence[float],
        *,
        asc: bool = True,
        distance_method: DistanceMethod | str,
        fetch_distance: bool,
    ) -> Self:
        qs = self.annotate_with_distance(
            query_vector, distance_method=distance_method, fetch_distance=fetch_distance
        ).order_by("distance")
        if not asc:
            qs = qs.reverse()
        return qs


class PgvectorEmbeddingManager(models.Manager.from_queryset(PgvectorEmbeddingQuerySet)):
    pass


class PgvectorEmbedding(models.Model):
    embedding = models.ForeignKey(
        "wagtail_vector_index.Embedding", on_delete=models.CASCADE, related_name="+"
    )
    vector = VectorField()
    embedding_output_dimensions = models.PositiveIntegerField(db_index=True)
    index_name = models.TextField()

    objects = PgvectorEmbeddingManager()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["embedding", "index_name", "embedding_output_dimensions"],
                name="unique_pgvector_embedding_per_index_and_dimensions",
            )
        ]
        indexes = [
            models.Index(fields=["index_name"]),
        ]
        verbose_name = _("pgvector embedding")
        # TODO: Determine if we need to add an indexes for the vector field.
        #       https://github.com/pgvector/pgvector-python/tree/master#django

    def __str__(self) -> str:
        return "pgvector embedding for {}".format(self.embedding)
