from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.fields import RichTextField
from wagtail.models import Page
from wagtail_vector_index.storage import get_storage_provider
from wagtail_vector_index.storage.base import VectorIndex
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsDocumentConverter,
    EmbeddingField,
    PageEmbeddableFieldsVectorIndexMixin,
    VectorIndexedMixin,
)
from wagtail_vector_index.storage.registry import registry


class ExampleModel(VectorIndexedMixin, models.Model):
    title = models.CharField(max_length=255)
    body = models.TextField()

    embedding_fields = [EmbeddingField("title", important=True), EmbeddingField("body")]

    def __str__(self):
        return self.title


class ExamplePage(VectorIndexedMixin, Page):
    body = RichTextField()

    content_panels = [*Page.content_panels, FieldPanel("body")]

    embedding_fields = [EmbeddingField("title", important=True), EmbeddingField("body")]


class DifferentPage(VectorIndexedMixin, Page):
    body = RichTextField()

    content_panels = [*Page.content_panels, FieldPanel("body")]

    embedding_fields = [EmbeddingField("title", important=True), EmbeddingField("body")]


def get_default_storage_mixin():
    provider = get_storage_provider("default")
    return provider.index_mixin


class MultiplePageVectorIndex(
    PageEmbeddableFieldsVectorIndexMixin, get_default_storage_mixin(), VectorIndex
):
    querysets = [ExamplePage.objects.all(), DifferentPage.objects.all()]  # type: ignore

    def get_converter(self):
        return EmbeddableFieldsDocumentConverter(Page)


registry.register_index(MultiplePageVectorIndex())
