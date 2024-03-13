from collections.abc import Generator, Iterable, MutableSequence
from typing import Self

from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models, transaction
from wagtail.models import Page
from wagtail.search.index import BaseField

from wagtail_vector_index.index.base import Document
from wagtail_vector_index.index.exceptions import IndexedTypeFromDocumentError
from wagtail_vector_index.index.model import (
    ModelVectorIndex,
    PageVectorIndex,
)

from .ai_utils.backends.base import BaseEmbeddingBackend
from .ai_utils.text_splitting.langchain import LangchainRecursiveCharacterTextSplitter
from .ai_utils.text_splitting.naive import NaiveTextSplitterCalculator


class Embedding(models.Model):
    """Stores an embedding for a model instance"""

    content_type = models.ForeignKey(
        ContentType, on_delete=models.CASCADE, related_name="+"
    )
    content_type_id: int
    base_content_type = models.ForeignKey(
        ContentType, on_delete=models.CASCADE, related_name="+"
    )
    object_id = models.CharField(
        max_length=255,
    )
    content_object = GenericForeignKey(
        "content_type", "object_id", for_concrete_model=False
    )
    vector = models.JSONField()
    content = models.TextField()

    def __str__(self):
        return f"Embedding for {self.object_id}"

    @classmethod
    def _get_base_content_type(cls, model_or_object):
        if parents := model_or_object._meta.get_parent_list():
            return ContentType.objects.get_for_model(
                parents[-1], for_concrete_model=False
            )
        else:
            return ContentType.objects.get_for_model(
                model_or_object, for_concrete_model=False
            )

    @classmethod
    def from_instance(cls, instance: models.Model) -> "Embedding":
        """Create an Embedding instance for a model instance"""
        content_type = ContentType.objects.get_for_model(instance)
        return Embedding(
            content_type=content_type,
            base_content_type=cls._get_base_content_type(instance),
            object_id=instance.pk,
        )

    @classmethod
    def get_for_instance(cls, instance: models.Model):
        """Get all Embedding instances that are related to a model instance"""
        content_type = ContentType.objects.get_for_model(instance)
        return Embedding.objects.filter(
            content_type=content_type, object_id=instance.pk
        )

    def to_document(self) -> Document:
        return Document(
            id=str(self.pk),
            vector=self.vector,
            metadata={
                "object_id": str(self.object_id),
                "content_type_id": str(self.content_type_id),
                "content": self.content,
            },
        )


class EmbeddingField(BaseField):
    """A field that can be used to specify which fields of a model should be used to generate embeddings"""

    def __init__(self, *args, important=False, **kwargs):
        self.important = important
        super().__init__(*args, **kwargs)


class VectorIndexedMixin(models.Model):
    """Mixin for Django models that make them conform to the VectorIndexable protocol and stores
    embeddings in an Embedding model"""

    embedding_fields = []
    embeddings = GenericRelation(
        Embedding, content_type_field="content_type", for_concrete_model=False
    )
    vector_index_class = None

    class Meta:
        abstract = True

    @classmethod
    def _get_embedding_fields(cls) -> list["EmbeddingField"]:
        embedding_fields = {
            (type(field), field.field_name): field for field in cls.embedding_fields
        }
        return list(embedding_fields.values())

    def _get_text_splitter(
        self, chunk_size: int
    ) -> LangchainRecursiveCharacterTextSplitter:
        length_calculator = NaiveTextSplitterCalculator()
        return LangchainRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=length_calculator.get_splitter_length,
        )

    def _get_split_content(self, *, chunk_size: int) -> list[str]:
        """Split the contents of a model instance's `embedding_fields` in to smaller chunks"""
        splittable_content = []
        important_content = []
        embedding_fields = self._meta.model._get_embedding_fields()

        for field in embedding_fields:
            value = field.get_value(self)
            if value is None:
                continue
            if isinstance(value, str):
                final_value = value
            else:
                final_value: str = "\n".join((str(v) for v in value))
            if field.important:
                important_content.append(final_value)
            else:
                splittable_content.append(final_value)

        text = "\n".join(splittable_content)
        important_text = "\n".join(important_content)
        splitter = self._get_text_splitter(chunk_size=chunk_size)
        return [f"{important_text}\n{text}" for text in splitter.split_text(text)]

    @classmethod
    def check(cls, **kwargs):
        """Extend model checks to include validation of embedding_fields in the
        same way that Wagtail's Indexed class does it."""
        errors = super().check(**kwargs)
        errors.extend(cls._check_embedding_fields(**kwargs))
        return errors

    @classmethod
    def _has_field(cls, name):
        try:
            cls._meta.get_field(name)
        except FieldDoesNotExist:
            return hasattr(cls, name)
        else:
            return True

    @classmethod
    def _check_embedding_fields(cls, **kwargs):
        errors = []
        for field in cls._get_embedding_fields():
            message = "{model}.embedding_fields contains non-existent field '{name}'"
            if not cls._has_field(field.field_name):
                errors.append(
                    checks.Warning(
                        message.format(model=cls.__name__, name=field.field_name),
                        obj=cls,
                        id="wagtailai.WA001",
                    )
                )
        return errors

    def _existing_embeddings_match(
        self, embeddings: Iterable[Embedding], splits: list[str]
    ) -> bool:
        """Determine whether the embeddings passed in match the text content passed in"""
        if not embeddings:
            return False

        embedding_content = {embedding.content for embedding in embeddings}

        return set(splits) == embedding_content

    @transaction.atomic
    def generate_embeddings(
        self, *, embedding_backend: BaseEmbeddingBackend
    ) -> list[Embedding]:
        """Use the AI backend to generate and store embeddings for this object"""
        splits = self._get_split_content(
            chunk_size=embedding_backend.config.token_limit
        )
        embeddings = Embedding.get_for_instance(self)

        # If the existing embeddings all match on content, we return them
        # without generating new ones
        if self._existing_embeddings_match(embeddings, splits):
            return list(embeddings)

        # Otherwise we delete all the existing embeddings and get new ones
        embeddings.delete()

        embedding_vectors = embedding_backend.embed(splits)
        generated_embeddings: MutableSequence[Embedding] = []
        for idx, returned_embedding in enumerate(embedding_vectors):
            split = splits[idx]
            embedding = Embedding.from_instance(self)
            embedding.vector = returned_embedding
            embedding.content = split
            embedding.save()
            generated_embeddings.append(embedding)

        return generated_embeddings

    def to_documents(
        self, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]:
        for embedding in self.generate_embeddings(embedding_backend=embedding_backend):
            yield embedding.to_document()

    @classmethod
    def from_document(cls, document) -> Self:
        if obj := cls.objects.filter(
            pk=document.metadata["object_id"],
            content_type=document.metadata["content_type_id"],
        ).first():
            return obj
        else:
            raise IndexedTypeFromDocumentError("No object found for document")

    @classmethod
    def bulk_to_documents(
        cls, objects, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]:
        # TODO: Implement a more efficient bulk embedding approach
        for object in objects:
            yield from object.to_documents(embedding_backend=embedding_backend)

    @classmethod
    def bulk_from_documents(cls, documents):
        # TODO: Implement a more efficient approach
        for document in documents:
            yield cls.from_document(document)

    @classmethod
    def get_vector_index(cls):
        """Get a vector index instance for this model"""

        # If the user has specified a custom `vector_index_class`, use that
        if cls.vector_index_class:
            index_cls = cls.vector_index_class
        # If the model is a Wagtail Page, use a special PageVectorIndex
        elif issubclass(cls, Page):
            index_cls = PageVectorIndex
        # Otherwise use the standard ModelVectorIndex
        else:
            index_cls = ModelVectorIndex

        return type(
            f"{cls.__name__}Index",
            (index_cls,),
            {
                "querysets": [cls.objects.all()],
                "object_type": cls,
            },
        )()
