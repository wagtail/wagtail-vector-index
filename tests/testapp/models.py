from typing import Sequence

from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.fields import RichTextField
from wagtail.models import Page
from wagtail_vector_index.storage.django import (
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

    content_panels = [*Page.content_panels, FieldPanel("body")]

    embedding_fields = [EmbeddingField("title", important=True), EmbeddingField("body")]


class FilmPage(VectorIndexedMixin, Page):
    description = models.TextField()

    content_panels = [*Page.content_panels, FieldPanel("description")]

    embedding_fields = [
        EmbeddingField("title", important=True),
        EmbeddingField("description"),
    ]

    @property
    def body(self):
        return self.description


class VideoGame(VectorIndexedMixin, models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()

    embedding_fields = [
        EmbeddingField("title", important=True),
        EmbeddingField("description"),
    ]

    def __str__(self):
        return self.title

    @property
    def body(self):
        return self.description


class AllMediaVectorIndex(EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex):
    def get_querysets(self) -> Sequence[models.QuerySet]:
        return [
            BookPage.objects.all(),
            FilmPage.objects.all(),
            VideoGame.objects.all(),
        ]


registry.register_index(AllMediaVectorIndex())
