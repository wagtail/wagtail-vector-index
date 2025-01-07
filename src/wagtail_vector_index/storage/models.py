import operator
from collections.abc import AsyncGenerator
from functools import reduce
from typing import TYPE_CHECKING, cast

from django.db import connection, models
from django.db.models import Q

if TYPE_CHECKING:
    from wagtail_vector_index.storage.conversion import DocumentFilter


class DocumentQuerySet(models.QuerySet):
    def for_key(self, object_key: str):
        if connection.vendor != "sqlite":
            return self.filter(object_keys__contains=[object_key])
        else:
            # SQLite doesn't support the __contains lookup for JSON fields
            # so we use icontains which just does a string search
            return self.filter(object_keys__icontains=object_key)

    async def afor_key(self, object_key: str) -> AsyncGenerator["Document", None]:
        if connection.vendor != "sqlite":
            async for doc in self.filter(object_keys__contains=[object_key]):
                yield doc
        else:
            # SQLite doesn't support the __contains lookup for JSON fields
            # so we use icontains which just does a string search
            async for doc in self.filter(object_keys__icontains=object_key):
                yield doc

    def for_keys(self, object_keys: list[str]):
        if not object_keys:
            return self.none()
        q_objs = [Q(object_keys__icontains=object_key) for object_key in object_keys]
        if not q_objs:
            return self
        return self.filter(reduce(operator.or_, q_objs))

    async def afor_keys(
        self, object_keys: list[str]
    ) -> AsyncGenerator["Document", None]:
        if not object_keys:
            return
        q_objs = [Q(object_keys__icontains=object_key) for object_key in object_keys]
        if q_objs:
            filtered_docs = self.filter(reduce(operator.or_, q_objs))
        else:
            filtered_docs = self
        async for doc in filtered_docs:
            yield doc

    def for_model_types(self, *model_types: type[models.Model]) -> "DocumentQuerySet":
        q_objs = [Q(object_keys__icontains=f"{mt._meta.label}:") for mt in model_types]
        return self.filter(reduce(operator.or_, q_objs))

    async def afor_model_types(
        self, *model_types: type[models.Model]
    ) -> AsyncGenerator["Document", None]:
        q_objs = [Q(object_keys__icontains=f"{mt._meta.label}:") for mt in model_types]
        async for doc in self.filter(reduce(operator.or_, q_objs)):
            yield doc

    def apply_filters(self, filters: list["DocumentFilter"]) -> "DocumentQuerySet":
        for filter in filters:
            self = filter.apply(self)
        return self

    @classmethod
    def as_manager(cls) -> "DocumentManager":
        return cast(DocumentManager, super().as_manager())


class DocumentManager(models.Manager["Document"]):
    # Workaround for typing issues
    def for_key(self, object_key: str) -> DocumentQuerySet: ...

    def for_keys(self, object_keys: list[str]) -> DocumentQuerySet: ...

    def afor_key(self, object_key: str) -> AsyncGenerator["Document", None]: ...

    def afor_keys(self, object_keys: list[str]) -> AsyncGenerator["Document", None]: ...

    def for_model_types(self, *model_types: type[models.Model]) -> DocumentQuerySet: ...

    def afor_model_types(
        self, *model_types: type[models.Model]
    ) -> AsyncGenerator["Document", None]: ...

    def apply_filters(self, filters: list["DocumentFilter"]) -> DocumentQuerySet: ...


class Document(models.Model):
    """Stores an embedding for an arbitrary chunk"""

    object_keys = models.JSONField(default=list)
    vector = models.JSONField()
    content = models.TextField()
    metadata = models.JSONField(default=dict)

    objects: DocumentManager = DocumentQuerySet.as_manager()

    def __str__(self):
        keys = ", ".join(self.object_keys)
        return f"Document for {keys}"

    @classmethod
    def from_keys(cls, object_keys: list[str]) -> "Document":
        """Create a Document instance for a list of object keys"""
        return Document(
            object_keys=object_keys,
        )
