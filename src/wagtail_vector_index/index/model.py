from typing import TYPE_CHECKING, Iterable, List

from django.apps import apps
from django.db import models
from wagtail.query import PageQuerySet

from .base import Document, VectorIndex
from .registry import registry

if TYPE_CHECKING:
    pass


class ModelVectorIndex(VectorIndex["VectorIndexedMixin"]):
    """A VectorIndex which indexes the results of querysets of VectorIndexedMixin models"""

    querysets: List[models.QuerySet]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_querysets(self) -> List[models.QuerySet]:
        return self.querysets

    def rebuild_index(self):
        """Before building an index, generate Embedding objects for everything in
        this index"""
        querysets = self._get_querysets()

        # TODO Rework - shouldn't need to pull in an AI backend here
        from wagtail_vector_index.ai import get_ai_backend

        ai_backend = get_ai_backend()

        for queryset in querysets:
            for instance in queryset:
                instance.generate_embeddings(ai_backend=ai_backend)

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
                            "content_type_id": embedding.content_type.pk,
                            "object_id": embedding.object_id,
                            "content": embedding.content,
                        },
                    )


class PageVectorIndex(ModelVectorIndex):
    """A model vector indexed for use with Wagtail pages that automatically
    restricts indexed models to live pages."""

    querysets: List[PageQuerySet]

    def _get_querysets(self) -> List[PageQuerySet]:
        qs_list = super()._get_querysets()
        return [qs.live() for qs in qs_list]


def register_indexed_models():
    """Discover and register all models that are a subclass of VectorIndexedMixin"""
    from wagtail_vector_index.models import VectorIndexedMixin

    indexed_models = [
        model
        for model in apps.get_models()
        if issubclass(model, VectorIndexedMixin) and not model._meta.abstract
    ]
    for model in indexed_models:
        registry.register()(model.get_vector_index().__class__)
