from collections.abc import Generator, Iterable, MutableSequence, Sequence
from typing import (
    ClassVar,
    Optional,
    TypeVar,
    cast,
)

from django.apps import apps
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models, transaction
from django.utils.functional import classproperty  # type: ignore
from wagtail.models import Page
from wagtail.query import PageQuerySet
from wagtail.search.index import BaseField

from wagtail_vector_index.ai_utils.backends.base import BaseEmbeddingBackend
from wagtail_vector_index.ai_utils.text_splitting.langchain import (
    LangchainRecursiveCharacterTextSplitter,
)
from wagtail_vector_index.ai_utils.text_splitting.naive import (
    NaiveTextSplitterCalculator,
)
from wagtail_vector_index.ai_utils.types import TextSplitterProtocol
from wagtail_vector_index.index import registry
from wagtail_vector_index.index.base import Document, VectorIndex
from wagtail_vector_index.index.exceptions import IndexedTypeFromDocumentError

""" Everything related to indexing Django models is in this file.

This includes:

- The Embedding Django model, which is used to store embeddings for model instances in the database
- The EmbeddableFieldsMixin, which is a mixin for Django models that lets user define which fields should be used to generate embeddings
- The EmbeddableFieldsVectorIndex, which is a VectorIndex that expects EmbeddableFieldsMixin models
- The EmbeddableFieldsDocumentConverter, which is a DocumentConverter that knows how to convert a model instance using the EmbeddableFieldsMixin protocol to and from a Document
"""


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
            vector=self.vector,
            embedding_pk=self.pk,
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


class EmbeddableFieldsMixin(models.Model):
    """Mixin for Django models that allows the user to specify which fields should be used to generate embeddings."""

    embedding_fields = []
    embeddings = GenericRelation(
        Embedding, content_type_field="content_type", for_concrete_model=False
    )

    class Meta:
        abstract = True

    @classmethod
    def _get_embedding_fields(cls) -> list["EmbeddingField"]:
        embedding_fields = {
            (type(field), field.field_name): field for field in cls.embedding_fields
        }
        return list(embedding_fields.values())

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


IndexedType = TypeVar("IndexedType")


class EmbeddableFieldsDocumentConverter:
    """Implementation of DocumentConverter that knows how to convert a model instance using the
    EmbeddableFieldsMixin to and from a Document.

    Stores and retrieves embeddings from an Embedding model."""

    def __init__(self, base_model: type[models.Model]):
        # The model that this converter will convert Documents back to
        self.base_model = base_model

    def _get_split_content(
        self, object: EmbeddableFieldsMixin, *, chunk_size: int
    ) -> list[str]:
        """Split the contents of a model instance's `embedding_fields` in to smaller chunks"""
        splittable_content = []
        important_content = []
        embedding_fields = object._meta.model._get_embedding_fields()

        for field in embedding_fields:
            value = field.get_value(object)
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
        splitter = self._get_text_splitter_class(chunk_size=chunk_size)
        return [f"{important_text}\n{text}" for text in splitter.split_text(text)]

    def _get_text_splitter_class(self, chunk_size: int) -> TextSplitterProtocol:
        length_calculator = NaiveTextSplitterCalculator()
        return LangchainRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=length_calculator.get_splitter_length,
        )

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
        self, object: EmbeddableFieldsMixin, *, embedding_backend: BaseEmbeddingBackend
    ) -> list[Embedding]:
        """Use the AI backend to generate and store embeddings for this object"""
        splits = self._get_split_content(
            object, chunk_size=embedding_backend.config.token_limit
        )
        embeddings = Embedding.get_for_instance(object)

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
            embedding = Embedding.from_instance(object)
            embedding.vector = returned_embedding
            embedding.content = split
            embedding.save()
            generated_embeddings.append(embedding)

        return generated_embeddings

    def to_documents(
        self, object: EmbeddableFieldsMixin, *, embedding_backend: BaseEmbeddingBackend
    ) -> Generator[Document, None, None]:
        for embedding in self.generate_embeddings(
            object, embedding_backend=embedding_backend
        ):
            yield embedding.to_document()

    def from_document(self, document: Document) -> models.Model:
        if obj := self.base_model.objects.filter(
            pk=document.metadata["object_id"],
            content_type=document.metadata["content_type_id"],
        ).first():
            return obj
        else:
            raise IndexedTypeFromDocumentError("No object found for document")

    def bulk_to_documents(
        self,
        objects: Iterable[EmbeddableFieldsMixin],
        *,
        embedding_backend: BaseEmbeddingBackend,
    ) -> Generator[Document, None, None]:
        # TODO: Implement a more efficient bulk embedding approach
        for object in objects:
            yield from self.to_documents(object, embedding_backend=embedding_backend)

    def bulk_from_documents(
        self, documents: Iterable[Document]
    ) -> Generator[models.Model, None, None]:
        # TODO: Implement a more efficient approach
        for document in documents:
            yield self.from_document(document)


class EmbeddableFieldsVectorIndex(VectorIndex):
    """A VectorIndex which indexes the results of querysets of EmbeddableFieldsMixin models"""

    querysets: ClassVar[Sequence[models.QuerySet]]

    def _get_querysets(self) -> Sequence[models.QuerySet]:
        return self.querysets

    def get_converter_class(self) -> type[EmbeddableFieldsDocumentConverter]:
        return EmbeddableFieldsDocumentConverter

    def get_converter(self) -> EmbeddableFieldsDocumentConverter:
        queryset_models = [qs.model for qs in self._get_querysets()]
        all_the_same = len(set(queryset_models)) == 1
        if not all_the_same:
            raise ValueError(
                "All querysets must be of the same model to use the default converter."
            )
        return self.get_converter_class()(queryset_models[0])

    def get_documents(self) -> Iterable[Document]:
        querysets = self._get_querysets()
        all_documents = []

        for queryset in querysets:
            instances = queryset.prefetch_related("embeddings")
            # We need to consume the generator here to ensure that the
            # Embedding models are created, even if it is not consumed
            # by the caller
            all_documents += list(
                self.get_converter().bulk_to_documents(
                    instances, embedding_backend=self.embedding_backend
                )
            )
        return all_documents


class PageEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    """A vector indexed for use with Wagtail pages that automatically
    restricts indexed models to live pages."""

    querysets: Sequence[PageQuerySet]

    def _get_querysets(self) -> list[PageQuerySet]:
        qs_list = super()._get_querysets()

        # Technically a manager instance, not a queryset, but we want to use the custom
        # methods.
        return [cast(PageQuerySet, qs).live() for qs in qs_list]


# ###########
# Classes related to automatic generation of indexes for models
# ###########


class GeneratedIndexMixin(models.Model):
    """Mixin for Django models that automatically generates registers a VectorIndex for the model.

    The model can still have custom VectorIndex classes registered if needed."""

    vector_index_class: ClassVar[Optional[type[VectorIndex]]] = None

    class Meta:
        abstract = True

    @classmethod
    def generated_index_class_name(cls):
        """Return the class name to be used for the index generated by this mixin"""
        return f"{cls.__name__}Index"

    @classmethod
    def build_vector_index_class(cls) -> type[VectorIndex]:
        """Build a VectorIndex class for this model"""

        # If the user has specified a custom `vector_index_class`, use that
        if cls.vector_index_class:
            index_cls = cls.vector_index_class
        # If the model is a Wagtail Page, use a special PageEmbeddableFieldsVectorIndex
        elif issubclass(cls, Page):
            index_cls = PageEmbeddableFieldsVectorIndex
        # Otherwise use the standard EmbeddableFieldsVectorIndex
        else:
            index_cls = EmbeddableFieldsVectorIndex

        return cast(
            type[VectorIndex],
            type(
                cls.generated_index_class_name(),
                (index_cls,),
                {
                    "querysets": [cls.objects.all()],
                },
            ),
        )

    @classproperty
    def vector_index(cls):
        """Get a vector index instance for this model"""

        return registry[cls.generated_index_class_name()]()


class VectorIndexedMixin(EmbeddableFieldsMixin, GeneratedIndexMixin, models.Model):
    """Model mixin which adds both the embeddable fields behaviour and the automatic index behaviour to a model."""

    class Meta:
        abstract = True


def register_indexed_models():
    """Discover and register all models that are a subclass of GeneratedIndexMixin."""
    indexed_models = [
        model
        for model in apps.get_models()
        if issubclass(model, GeneratedIndexMixin) and not model._meta.abstract
    ]
    for model in indexed_models:
        registry.register()(model.build_vector_index_class())
