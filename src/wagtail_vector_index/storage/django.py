import logging
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Generator,
    Iterable,
    Sequence,
)
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Optional,
    Type,
    TypeAlias,
    cast,
)

from asgiref.sync import sync_to_async
from django.apps import apps
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
    DocumentConverter,
    DocumentRetrievalVectorIndexMixinProtocol,
    FromDocumentOperator,
    ObjectChunkerOperator,
    ToDocumentOperator,
    VectorIndex,
)
from wagtail_vector_index.storage.exceptions import IndexedTypeFromDocumentError
from wagtail_vector_index.storage.models import Document

logger = logging.getLogger(__name__)

""" Everything related to indexing Django models is in this file.

This includes:

- The Embedding Django model, which is used to store embeddings for model instances in the database
- The EmbeddableFieldsMixin, which is a mixin for Django models that lets user define which fields should be used to generate embeddings
- The EmbeddableFieldsVectorIndexMixin, which is a VectorIndex mixin that expects EmbeddableFieldsMixin models
- The EmbeddableFieldsDocumentConverter, which is a DocumentConverter that knows how to convert a model instance using the EmbeddableFieldsMixin protocol to and from a Document
"""

ModelLabel: TypeAlias = str
ObjectId: TypeAlias = str


# If `batched` is not available (Python < 3.12), provide a fallback implementation
try:
    from itertools import batched  # type: ignore
except ImportError:

    def batched(iterable, n):
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


class ModelKey(str):
    """A unique identifier for a model instance.

    The string is of the form "<model_label>:<object_id>". This can be used as the object_key
    for Documents.
    """

    @classmethod
    def from_instance(cls, instance: models.Model) -> "ModelKey":
        return cls(f"{instance._meta.label}:{instance.pk}")

    @property
    def model_label(self) -> ModelLabel:
        return self.split(":")[0]

    @property
    def object_id(self) -> ObjectId:
        return self.split(":")[1]


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
        for field_ in cls._get_embedding_fields():
            message = "{model}.embedding_fields contains non-existent field '{name}'"
            if not cls._has_field(field_.field_name):
                errors.append(
                    checks.Warning(
                        message.format(model=cls.__name__, name=field_.field_name),
                        obj=cls,
                        id="wagtailai.WA001",
                    )
                )
        return errors


class ModelFromDocumentOperator(FromDocumentOperator[models.Model]):
    """A class that can convert Documents into model instances"""

    def from_document(self, document: Document) -> models.Model:
        # Use the first key in the list, which is the most specific model class
        key = ModelKey(document.object_keys[0])
        model_class = self._model_class_from_label(key.model_label)
        try:
            return model_class.objects.filter(pk=key.object_id).get()
        except model_class.DoesNotExist as e:
            raise IndexedTypeFromDocumentError("No object found for document") from e

    def bulk_from_documents(
        self, documents: Sequence[Document]
    ) -> Generator[models.Model, None, None]:
        keys_by_model_label = self._get_keys_by_model_label(documents)
        objects_by_key = self._get_models_by_key(keys_by_model_label)

        yield from self._get_deduplicated_objects_generator(
            documents=documents, objects_by_key=objects_by_key
        )

    async def abulk_from_documents(
        self, documents: Sequence[Document]
    ) -> AsyncGenerator[models.Model, None]:
        """A copy of `bulk_from_documents`, but async"""
        keys_by_model_label = self._get_keys_by_model_label(documents)
        objects_by_key = await self._aget_models_by_key(keys_by_model_label)

        # N.B. `yield from`  cannot be used in async functions, so we have to use a loop
        for object_from_document in self._get_deduplicated_objects_generator(
            documents=documents, objects_by_key=objects_by_key
        ):
            yield object_from_document

    @staticmethod
    def _model_class_from_label(label: ModelLabel) -> type[models.Model]:
        model_class = apps.get_model(label)

        if model_class is None:
            raise ValueError(f"Failed to find model class for {label!r}")

        return model_class

    @staticmethod
    def _get_keys_by_model_label(
        documents: Sequence[Document],
    ) -> dict[ModelLabel, list[ModelKey]]:
        keys_by_model_label = defaultdict(list)
        for doc in documents:
            key = ModelKey(doc.object_keys[0])
            keys_by_model_label[key.model_label].append(key)
        return keys_by_model_label

    @staticmethod
    def _get_deduplicated_objects_generator(
        *, documents: Sequence[Document], objects_by_key: dict[ModelKey, models.Model]
    ) -> Generator[models.Model, None, None]:
        seen_keys = set()  # de-dupe as we go
        for doc in documents:
            key = ModelKey(doc.object_keys[0])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield objects_by_key[key]

    @staticmethod
    def _get_models_by_key(keys_by_model_label: dict) -> dict[ModelKey, models.Model]:
        """
        ModelKey keys are required to reliably map data
        from multiple models. This function loads the models from the database
        and groups them by such a key.
        """
        objects_by_key: dict[ModelKey, models.Model] = {}
        for model_label, keys in keys_by_model_label.items():
            model_class = ModelFromDocumentOperator._model_class_from_label(model_label)
            model_objects = model_class.objects.filter(
                pk__in=[key.object_id for key in keys]
            )
            objects_by_key.update(
                {ModelKey.from_instance(obj): obj for obj in model_objects}
            )
        return objects_by_key

    @staticmethod
    async def _aget_models_by_key(
        keys_by_model_label: dict,
    ) -> dict[ModelKey, models.Model]:
        """
        Same as `_get_models_by_key`, but async.
        """
        objects_by_key: dict[ModelKey, models.Model] = {}
        for model_label, keys in keys_by_model_label.items():
            model_class = ModelFromDocumentOperator._model_class_from_label(model_label)
            model_objects = model_class.objects.filter(
                pk__in=[key.object_id for key in keys]
            )
            objects_by_key.update(
                {ModelKey.from_instance(obj): obj async for obj in model_objects}
            )
        return objects_by_key


@dataclass
class PreparedObject:
    """A class that represents a model instance and its chunks - used to persist object metadata and allow batch operations"""

    key: ModelKey
    object: models.Model
    chunks: list[str]
    embedding_vectors: list[list[float]] | None = None
    existing_documents: list[Document] | None = None
    new_documents: list[Document] = field(default_factory=list)

    @property
    def documents(self) -> list[Document]:
        return self.new_documents or self.existing_documents or []

    @property
    def needs_updating(self) -> bool:
        """Determine whether the embeddings need to be updated for this object"""
        if not self.existing_documents:
            return True

        document_content = {document.content for document in self.existing_documents}

        return set(self.chunks) != document_content


@dataclass
class PreparedObjectCollection:
    """A collection of PreparedObjects that handles bulk operations like chunk mapping and embedding"""

    objects: list[PreparedObject]

    def __iter__(self) -> Iterator[PreparedObject]:
        """Make the collection iterable, yielding PreparedObjects"""
        return iter(self.objects)

    @classmethod
    def prepare_objects(
        cls,
        objects: Iterable[models.Model],
        *,
        chunker_operator: ObjectChunkerOperator,
        embedding_backend: BaseEmbeddingBackend,
    ) -> "PreparedObjectCollection":
        """Create a PreparedObjectCollection from a list of model instances"""
        prepared_objects = []
        all_keys = []

        for object in objects:
            key = ModelKey.from_instance(object)
            chunks = list(
                chunker_operator.chunk_object(
                    object, chunk_size=embedding_backend.config.token_limit
                )
            )
            prepared_objects.append(
                PreparedObject(key=key, object=object, chunks=chunks)
            )
            all_keys.append(key)

        existing_documents = Document.objects.for_keys(all_keys)
        existing_documents_by_key = cls._group_documents_by_object_key(
            existing_documents
        )

        for object in prepared_objects:
            object.existing_documents = existing_documents_by_key[object.key]

        return cls(objects=prepared_objects)

    @staticmethod
    def _group_documents_by_object_key(documents) -> dict[ModelKey, list[Document]]:
        """Group documents by their object key"""
        documents_by_object_key = defaultdict(list)
        for document in documents:
            documents_by_object_key[document.object_keys[0]].append(document)
        return documents_by_object_key

    @property
    def objects_by_key(self) -> dict[ModelKey, PreparedObject]:
        return {obj.key: obj for obj in self.objects}

    @property
    def objects_needing_update(self) -> list[PreparedObject]:
        """Return list of objects that need their embeddings updated"""
        return [obj for obj in self.objects if obj.needs_updating]

    def get_all_chunks(self) -> list[str]:
        """Get all chunks from objects needing updates"""
        return [chunk for obj in self.objects_needing_update for chunk in obj.chunks]

    def get_chunk_mapping(self) -> list[ModelKey]:
        """Create a mapping of chunk indices to object keys"""
        return [obj.key for obj in self.objects_needing_update for _ in obj.chunks]

    @staticmethod
    def _keys_for_instance(instance: models.Model) -> list[ModelKey]:
        """Get keys for all the parent classes and the object itself in MRO order"""
        parent_classes = instance._meta.get_parent_list()
        keys = [ModelKey(f"{cls._meta.label}:{instance.pk}") for cls in parent_classes]
        keys = [ModelKey.from_instance(instance), *keys]
        return keys

    def prepare_new_documents(self, embedding_vectors: Iterable[list[float]]):
        """Prepare (but don't save) new documents for objects that need updating with new embeddings"""
        if not embedding_vectors:
            return

        chunk_mapping = self.get_chunk_mapping()
        all_chunks = self.get_all_chunks()

        # Group embeddings by object
        embeddings_by_key: dict[ModelKey, list[tuple[int, list[float]]]] = defaultdict(
            list
        )
        for idx, embedding in enumerate(embedding_vectors):
            object_key = chunk_mapping[idx]
            embeddings_by_key[object_key].append((idx, embedding))

        # Create new documents for each object
        for object_key, embeddings in embeddings_by_key.items():
            prepared_obj = self.objects_by_key[object_key]
            for idx, embedding in embeddings:
                chunk = all_chunks[idx]
                all_keys = self._keys_for_instance(prepared_obj.object)
                prepared_obj.new_documents.append(
                    Document(
                        object_keys=all_keys,
                        vector=embedding,
                        content=chunk,
                    )
                )

        print([obj.new_documents for obj in self.objects])


class ModelToDocumentOperator(ToDocumentOperator[models.Model]):
    """A class that can generate Documents from model instances"""

    def __init__(self, object_chunker_operator_class: Type[ObjectChunkerOperator]):
        self.object_chunker_operator = object_chunker_operator_class()

    @transaction.atomic
    def update_documents(
        self,
        collection: PreparedObjectCollection,
    ):
        """Replace the current Documents for all objects that have new documents to save"""
        replaced_keys = [str(obj.key) for obj in collection if obj.new_documents]
        if replaced_keys:
            Document.objects.for_keys(replaced_keys).delete()
            Document.objects.bulk_create(
                chain(*[obj.new_documents for obj in collection if obj.new_documents])
            )

    def _update_object_collection_with_new_documents(
        self,
        collection: PreparedObjectCollection,
        embedding_backend: BaseEmbeddingBackend,
    ):
        objects_to_rebuild = collection.objects_needing_update

        if not objects_to_rebuild:
            return list(
                chain(
                    *[
                        obj.existing_documents
                        for obj in collection
                        if obj.existing_documents
                    ]
                )
            )

        # Get embeddings for all chunks that need updating
        all_chunks = collection.get_all_chunks()
        embedding_vectors = list(embedding_backend.embed(all_chunks))

        # Apply the embeddings to create new documents
        collection.prepare_new_documents(embedding_vectors)

    # Helper methods for bulk document generation
    def _delete_existing_documents(self, *, documents_by_object):
        existing_documents = Document.objects.for_keys(list(documents_by_object.keys()))
        existing_documents.delete()

    async def _adelete_existing_documents(self, *, documents_by_object):
        existing_documents = Document.objects.for_keys(list(documents_by_object.keys()))
        await existing_documents.adelete()

    def _to_documents_batch(
        self, objects: Iterable[models.Model], embedding_backend: BaseEmbeddingBackend
    ):
        collection = PreparedObjectCollection.prepare_objects(
            objects=objects,
            chunker_operator=self.object_chunker_operator,
            embedding_backend=embedding_backend,
        )
        self._update_object_collection_with_new_documents(collection, embedding_backend)
        self.update_documents(collection)
        yield from [doc for obj in collection for doc in obj.documents]

    # Interface methods
    def to_documents(
        self,
        objects: Iterable[models.Model],
        *,
        embedding_backend: BaseEmbeddingBackend,
        batch_size: int = 100,
    ) -> Generator[Document, None, None]:
        batches = list(batched(objects, batch_size))
        for idx, batch in enumerate(batches):
            logger.info(f"Generating documents for batch {idx + 1} of {len(batches)}")
            for document in self._to_documents_batch(
                batch, embedding_backend=embedding_backend
            ):
                yield document

    async def _ato_documents_batch(
        self, objects: Iterable[models.Model], embedding_backend: BaseEmbeddingBackend
    ):
        collection = await sync_to_async(PreparedObjectCollection.prepare_objects)(
            objects=objects,
            chunker_operator=self.object_chunker_operator,
            embedding_backend=embedding_backend,
        )
        await self._aupdate_object_collection_with_new_documents(
            collection, embedding_backend
        )
        # Using sync_to_async to ensure the update can happen in a transaction
        await sync_to_async(self.update_documents)(collection)
        return [doc for obj in collection.objects for doc in obj.documents]

    async def _aupdate_object_collection_with_new_documents(
        self,
        collection: PreparedObjectCollection,
        embedding_backend: BaseEmbeddingBackend,
    ):
        """Async version of _update_object_collection_with_new_documents"""
        objects_to_rebuild = collection.objects_needing_update

        if not objects_to_rebuild:
            return list(
                chain(
                    *[
                        obj.existing_documents
                        for obj in collection
                        if obj.existing_documents
                    ]
                )
            )

        # Get embeddings for all chunks that need updating
        all_chunks = collection.get_all_chunks()
        embedding_vectors = await embedding_backend.aembed(all_chunks)

        # Apply the embeddings to create new documents
        collection.prepare_new_documents(embedding_vectors)

    async def ato_documents(
        self,
        objects: Iterable[models.Model],
        *,
        embedding_backend: BaseEmbeddingBackend,
        batch_size: int = 100,
    ) -> AsyncGenerator[Document, None]:
        batches = list(batched(objects, batch_size))
        for idx, batch in enumerate(batches):
            logger.info(f"Generating documents for batch {idx + 1} of {len(batches)}")
            for document in await self._ato_documents_batch(
                batch, embedding_backend=embedding_backend
            ):
                yield document


class EmbeddableFieldsObjectChunkerOperator(
    ObjectChunkerOperator[EmbeddableFieldsMixin]
):
    def chunk_object(
        self, object: EmbeddableFieldsMixin, *, chunk_size: int
    ) -> list[str]:
        """Split the contents of a model instance's `embedding_fields` in to smaller chunks"""
        splittable_content = []
        important_content = []
        embedding_fields = object._meta.model._get_embedding_fields()

        for field_ in embedding_fields:
            value = field_.get_value(object)
            if value is None:
                continue
            if isinstance(value, str):
                final_value = value
            else:
                final_value: str = "\n".join((str(v) for v in value))
            if field_.important:
                important_content.append(final_value)
            else:
                splittable_content.append(final_value)

        text = "\n".join(splittable_content)
        important_text = "\n".join(important_content)
        splitter = self._get_text_splitter_class(chunk_size=chunk_size)
        return [f"{important_text}\n{text}" for text in splitter.split_text(text)]

    @staticmethod
    def _get_text_splitter_class(chunk_size: int) -> TextSplitterProtocol:
        length_calculator = NaiveTextSplitterCalculator()
        return LangchainRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=length_calculator.get_splitter_length,
        )


class EmbeddableFieldsDocumentConverter(DocumentConverter):
    """Implementation of DocumentConverter that knows how to convert a model instance using the
    EmbeddableFieldsMixin to and from a Document.
    """

    to_document_operator_class = ModelToDocumentOperator
    from_document_operator_class = ModelFromDocumentOperator
    object_chunker_operator_class = EmbeddableFieldsObjectChunkerOperator


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
            # We need to consume the generator here to ensure that the
            # Embedding models are created, even if it is not consumed
            # by the caller
            all_documents += list(
                self.get_converter().to_documents(
                    queryset, embedding_backend=self.get_embedding_backend()
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

    e.g. If WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS includes a provider with alias "default" referencing the PgvectorStorageProvider,
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
