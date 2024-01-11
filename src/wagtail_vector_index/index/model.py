from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from django.apps import apps
from django.db import models
from wagtail.query import PageQuerySet

from .base import Document, VectorIndex
from .registry import registry

if TYPE_CHECKING:
    from wagtail_vector_index.models import VectorIndexedMixin  # noqa: F401


class ModelVectorIndex(VectorIndex["VectorIndexedMixin"]):
    """A VectorIndex which indexes the results of querysets of VectorIndexedMixin models"""

    querysets: Sequence[models.QuerySet]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_querysets(self) -> Sequence[models.QuerySet]:
        return self.querysets

    def rebuild_index(self):
        """Before building an index, generate Embedding objects for everything in
        this index"""
        querysets = self._get_querysets()

        for queryset in querysets:
            for instance in queryset:
                instance.generate_embeddings(embedding_backend=self.embedding_backend)

        return super().rebuild_index()

    def get_documents(self) -> Iterable[Document]:
        querysets = self._get_querysets()

        for queryset in querysets:
            for instance in queryset.prefetch_related("embeddings"):
                for embedding in instance.embeddings.all():
                    yield Document(
                        id=embedding.pk,
                        vector=embedding.vector,
                        metadata={
                            "content_type_id": embedding.content_type_id,
                            "object_id": embedding.object_id,
                            "content": embedding.content,
                        },
                    )


class PageVectorIndex(ModelVectorIndex):
    """A model vector indexed for use with Wagtail pages that automatically
    restricts indexed models to live pages."""

    querysets: Sequence[PageQuerySet]

    def _get_querysets(self) -> list[PageQuerySet]:
        qs_list = super()._get_querysets()

        # Technically a manager instance, not a queryset, but we want to use the custom
        # methods.
        return [cast(PageQuerySet, qs).live() for qs in qs_list]


def register_indexed_models():
    """Discover and register all models that are a subclass of VectorIndexedMixin"""
    from wagtail_vector_index.models import VectorIndexedMixin  # noqa: F811

    indexed_models = [
        model
        for model in apps.get_models()
        if issubclass(model, VectorIndexedMixin) and not model._meta.abstract
    ]
    for model in indexed_models:
        registry.register()(model.get_vector_index().__class__)
