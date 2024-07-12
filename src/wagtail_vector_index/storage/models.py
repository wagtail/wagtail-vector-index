from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Generator,
    Iterable,
    MutableSequence,
    Sequence,
)
from typing import TYPE_CHECKING, ClassVar, Optional, TypeVar, cast

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
from wagtail_vector_index.storage import get_storage_provider, registry
from wagtail_vector_index.storage.base import (
    Document,
    DocumentRetrievalVectorIndexMixinProtocol,
    VectorIndex,
)
from wagtail_vector_index.storage.exceptions import IndexedTypeFromDocumentError

""" Everything related to indexing Django models is in this file.

This includes:

- The Embedding Django model, which is used to store embeddings for model instances in the database
- The EmbeddableFieldsMixin, which is a mixin for Django models that lets user define which fields should be used to generate embeddings
- The EmbeddableFieldsVectorIndexMixin, which is a VectorIndex mixin that expects EmbeddableFieldsMixin models
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


# ###########
# Classes that allow users to automatically generate documents from their models based on fields specified
# ###########


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


class DocumentToModelMixin:
    """A mixin for DocumentConverter classes that need to efficiently convert Documents
    into model instances of the relevant type.
    """

    @staticmethod
    def _model_class_from_ctid(id: str) -> type[models.Model]:
        ct = ContentType.objects.get_for_id(int(id))
        model_class = ct.model_class()
        if model_class is None:
            raise ValueError(f"Failed to find model class for {ct!r}")
        return model_class

    @classmethod
    async def _amodel_class_from_ctid(cls, id: str) -> type[models.Model]:
        ct = await cls._aget_content_type_for_id(int(id))
        model_class = ct.model_class()
        if model_class is None:
            raise ValueError(f"Failed to find model class for {ct!r}")
        return model_class

    def from_document(self, document: Document) -> models.Model:
        model_class = self._model_class_from_ctid(document.metadata["content_type_id"])
        try:
            return model_class.objects.filter(pk=document.metadata["object_id"]).get()
        except model_class.DoesNotExist as e:
            raise IndexedTypeFromDocumentError("No object found for document") from e

    def bulk_from_documents(
        self, documents: Iterable[Document]
    ) -> Generator[models.Model, None, None]:
        # Force evaluate generators to allow value to be reused
        documents = tuple(documents)

        ids_by_content_type: dict[str, list[str]] = defaultdict(list)
        for doc in documents:
            ids_by_content_type[doc.metadata["content_type_id"]].append(
                doc.metadata["object_id"]
            )

        # NOTE: (content_type_id, object_id) combo keys are required to
        # reliably map data from multiple models
        objects_by_key: dict[tuple[str, str], models.Model] = {}
        for content_type_id, ids in ids_by_content_type.items():
            model_class = self._model_class_from_ctid(content_type_id)
            model_objects = model_class.objects.filter(pk__in=ids)
            objects_by_key.update(
                {(content_type_id, str(obj.pk)): obj for obj in model_objects}
            )

        seen_keys = set()  # de-dupe as we go
        for doc in documents:
            key = (doc.metadata["content_type_id"], doc.metadata["object_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield objects_by_key[key]

    async def abulk_from_documents(
        self, documents: Iterable[Document]
    ) -> AsyncGenerator[models.Model, None, None]:
        """A copy of `bulk_from_documents`, but async"""
        # Force evaluate generators to allow value to be reused
        documents = tuple(documents)

        ids_by_content_type: dict[str, list[str]] = defaultdict(list)
        for doc in documents:
            ids_by_content_type[doc.metadata["content_type_id"]].append(
                doc.metadata["object_id"]
            )

        # NOTE: (content_type_id, object_id) combo keys are required to
        # reliably map data from multiple models
        objects_by_key: dict[tuple[str, str], models.Model] = {}
        for content_type_id, ids in ids_by_content_type.items():
            model_class = await self._amodel_class_from_ctid(content_type_id)
            model_objects = model_class.objects.filter(pk__in=ids)
            objects_by_key.update(
                {(content_type_id, str(obj.pk)): obj async for obj in model_objects}
            )

        seen_keys = set()  # de-dupe as we go
        for doc in documents:
            key = (doc.metadata["content_type_id"], doc.metadata["object_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield objects_by_key[key]

    @staticmethod
    async def _aget_content_type_for_id(id: int) -> ContentType:
        """
        Same as `ContentTypeManager.get_for_id`, but async.
        """
        manager = ContentType.objects
        try:
            ct = manager._cache[manager.db][id]
        except KeyError:
            ct = await manager.aget(pk=id)
            manager._add_to_cache(manager.db, ct)
        return ct


class EmbeddableFieldsDocumentConverter(DocumentToModelMixin):
    """Implementation of DocumentConverter that knows how to convert a model instance using the
    EmbeddableFieldsMixin to and from a Document.

    Stores and retrieves embeddings from an Embedding model."""

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

    def bulk_to_documents(
        self,
        objects: Iterable[EmbeddableFieldsMixin],
        *,
        embedding_backend: BaseEmbeddingBackend,
    ) -> Generator[Document, None, None]:
        # TODO: Implement a more efficient bulk embedding approach
        for object in objects:
            yield from self.to_documents(object, embedding_backend=embedding_backend)


# ###########
# VectorIndex mixins which add model-specific behaviour
# ###########

if TYPE_CHECKING:
    MixinBase = DocumentRetrievalVectorIndexMixinProtocol
else:
    MixinBase = object


class EmbeddableFieldsVectorIndexMixin(MixinBase):
    """A Mixin for VectorIndex which indexes the results of querysets of EmbeddableFieldsMixin models"""

    querysets: ClassVar[Sequence[models.QuerySet]]

    def _get_querysets(self) -> Sequence[models.QuerySet]:
        return self.querysets

    def get_converter_class(self) -> type[EmbeddableFieldsDocumentConverter]:
        return EmbeddableFieldsDocumentConverter

    def get_converter(self) -> EmbeddableFieldsDocumentConverter:
        return self.get_converter_class()()

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
                    instances, embedding_backend=self.get_embedding_backend()
                )
            )
        return all_documents


class PageEmbeddableFieldsVectorIndexMixin(EmbeddableFieldsVectorIndexMixin):
    """A mixin for VectorIndex for use with Wagtail pages that automatically
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


def camel_case(snake_str: str):
    """Convert a snake_case string to CamelCase"""
    parts = snake_str.split("_")
    return "".join(*map(str.title, parts))


def build_vector_index_base_for_storage_provider(
    storage_provider_alias: str = "default",
):
    """Build a VectorIndex base class for a given storage provider alias.

    e.g. If WAGATAIL_VECTOR_INDEX_STORAGE_PROVIDERS includes a provider with alias "default" referencing the PgvectorStorageProvider,
    this function will return a class that is a subclass of PgvectorIndexMixin and VectorIndex.
    """

    storage_provider = get_storage_provider(storage_provider_alias)
    alias_camel = camel_case(storage_provider_alias)
    return type(
        f"{alias_camel}VectorIndex", (storage_provider.index_mixin, VectorIndex), {}
    )


# A VectorIndex built from whatever mixin belongs to the storage provider with the "default" alias
DefaultStorageVectorIndex = build_vector_index_base_for_storage_provider("default")


class GeneratedIndexMixin(models.Model):
    """Mixin for Django models that automatically generates and registers a VectorIndex for the model.

    The model can still have custom VectorIndex classes registered if needed."""

    vector_index_class: ClassVar[Optional[type[VectorIndex]]] = None

    class Meta:
        abstract = True

    @classmethod
    def generated_index_class_name(cls):
        """Return the class name to be used for the index generated by this mixin"""
        return f"{cls.__name__}Index"

    @classmethod
    def build_vector_index(cls) -> VectorIndex:
        """Build a VectorIndex instance for this model"""

        class_list = ()
        # If the user has specified a custom `vector_index_class`, use that
        if cls.vector_index_class:
            class_list = (cls.vector_index_class,)
        else:
            storage_provider = get_storage_provider("default")
            base_cls = VectorIndex
            storage_mixin_cls = storage_provider.index_mixin
            # If the model is a Wagtail Page, use a special PageEmbeddableFieldsVectorIndexMixin
            if issubclass(cls, Page):
                mixin_cls = PageEmbeddableFieldsVectorIndexMixin
            # Otherwise use the standard EmbeddableFieldsVectorIndexMixin
            else:
                mixin_cls = EmbeddableFieldsVectorIndexMixin
            class_list = (
                mixin_cls,
                storage_mixin_cls,
                base_cls,
            )

        return cast(
            type[VectorIndex],
            type(
                cls.generated_index_class_name(),
                class_list,
                {
                    "querysets": [cls.objects.all()],
                },
            ),
        )()

    @classproperty
    def vector_index(cls):
        """Get a vector index instance for this model"""

        return registry[cls.generated_index_class_name()]


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
        registry.register_index(model.build_vector_index())
