from typing import Self

from django.db import models
from django.utils.translation import gettext as _
from pgvector.django import VectorField


class PgvectorEmbeddingQuerySet(models.QuerySet["PgvectorEmbedding"]):
    def index(self, index_name: str) -> Self:
        return self.filter(index_name=index_name)


class PgvectorEmbedding(models.Model):
    embedding = models.ForeignKey(
        "wagtail_vector_index.Embedding", on_delete=models.CASCADE, related_name="+"
    )
    vector = VectorField()
    index_name = models.TextField()

    objects = PgvectorEmbeddingQuerySet.as_manager()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["embedding", "index_name"],
                name="unique_pgvector_embedding_per_index",
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
