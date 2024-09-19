from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.fields import RichTextField
from wagtail.models import Page
from wagtail_vector_index.storage.models import (
    DefaultStorageVectorIndex,
    EmbeddableFieldsVectorIndexMixin,
    EmbeddingField,
    VectorIndexedMixin,
)
from wagtail_vector_index.storage.registry import registry


class MediaIndexPage(Page):
    pass


class BookPage(VectorIndexedMixin, Page):
    body = RichTextField()

    embedding_fields = [EmbeddingField("title", important=True), EmbeddingField("body")]


class FilmPage(VectorIndexedMixin, Page):
    description = models.TextField()

    content_panels = [*Page.content_panels, FieldPanel("description")]

    embedding_fields = [
        EmbeddingField("title", important=True),
        EmbeddingField("description"),
    ]


class VideoGame(VectorIndexedMixin, models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()

    embedding_fields = [
        EmbeddingField("title", important=True),
        EmbeddingField("description"),
    ]

    def __str__(self):
        return self.title


class AllMediaVectorIndex(EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex):
    querysets = [
        BookPage.objects.all(),
        FilmPage.objects.all(),
        VideoGame.objects.all(),
    ]  # type: ignore


registry.register_index(AllMediaVectorIndex())
