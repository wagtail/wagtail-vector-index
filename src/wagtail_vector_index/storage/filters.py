from typing import Protocol

from django.db.models import Model, QuerySet

from wagtail_vector_index.storage.django import ModelKey
from wagtail_vector_index.storage.models import DocumentQuerySet


class DocumentFilter(Protocol):
    """Protocol for a class that can filter a list of documents"""

    def apply(self, documents: DocumentQuerySet) -> DocumentQuerySet: ...


class QuerySetFilter(DocumentFilter):
    """Filter a DocumentQuerySet to only include documents for objects in a given QuerySet"""

    def __init__(self, queryset: QuerySet):
        self.queryset = queryset

    def apply(self, documents: DocumentQuerySet) -> DocumentQuerySet:
        keys = [str(ModelKey.from_instance(obj)) for obj in self.queryset]
        return documents.for_keys(keys)


class ObjectTypeFilter(DocumentFilter):
    """Filter a DocumentQuerySet to only include documents for objects of specific model types"""

    def __init__(self, *model_types: type[Model]):
        self.model_types = model_types

    def apply(self, documents: DocumentQuerySet) -> DocumentQuerySet:
        return documents.for_model_types(*self.model_types)
