import operator
from functools import reduce
from typing import cast

from django.db import connection, models
from django.db.models import Q


class DocumentQuerySet(models.QuerySet):
    def for_key(self, object_key: str):
        if connection.vendor != "sqlite":
            return self.filter(object_keys__contains=[object_key])
        else:
            # SQLite doesn't support the __contains lookup for JSON fields
            # so we use icontains which just does a string search
            return self.filter(object_keys__icontains=object_key)

    def for_keys(self, object_keys: list[str]):
        q_objs = [Q(object_keys__icontains=object_key) for object_key in object_keys]
        return self.filter(reduce(operator.or_, q_objs))

    @classmethod
    def as_manager(cls) -> "DocumentManager":
        return cast(DocumentManager, super().as_manager())


class DocumentManager(models.Manager["Document"]):
    # Workaround for typing issues
    def for_key(self, object_key: str) -> DocumentQuerySet: ...

    def for_keys(self, object_keys: list[str]) -> DocumentQuerySet: ...


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
