import operator
from functools import reduce

from django.db import connection, models
from django.db.models import Q


class DocumentManager(models.Manager):
    def for_key(self, object_key: str):
        if connection.vendor != "sqlite":
            return self.filter(object_keys__contains=[object_key])
        else:
            # SQLite doesn't support the __contains lookup for JSON fields
            # We need to use a different approach for SQLite
            return self.filter(object_keys__icontains=object_key)

    def for_keys(self, object_keys: list[str]):
        if connection.vendor != "sqlite":
            return self.filter(object_keys__contains=object_keys)
        else:
            # SQLite doesn't support the __contains lookup for JSON fields
            # We need to use a different approach for SQLite
            q_objs = [
                Q(object_keys__icontains=object_key) for object_key in object_keys
            ]
            return self.filter(reduce(operator.or_, q_objs))


class Document(models.Model):
    """Stores an embedding for an arbitrary chunk"""

    object_keys = models.JSONField(default=list)
    vector = models.JSONField()
    content = models.TextField()
    metadata = models.JSONField(default=dict)

    objects: DocumentManager = DocumentManager()

    def __str__(self):
        keys = ", ".join(self.object_keys)
        return f"Document for {keys}"

    @classmethod
    def from_keys(cls, object_keys: list[str]) -> "Document":
        """Create a Document instance for a list of object keys"""
        return Document(
            object_keys=object_keys,
        )
