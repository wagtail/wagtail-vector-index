from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from wagtail.search.index import BaseField

from wagtail_vector_index.ai_utils.text_splitting.langchain import (
    LangchainRecursiveCharacterTextSplitter,
)
from wagtail_vector_index.ai_utils.text_splitting.naive import (
    NaiveTextSplitterCalculator,
)
from wagtail_vector_index.ai_utils.types import TextSplitterProtocol


class ChunkableProtocol:
    """Protocol for objects that can be chunked into smaller chunks"""

    def get_chunks(self, *, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
        return []


class EmbeddingField(BaseField):
    """A field that can be used to specify which fields of a model should be used to generate embeddings"""

    def __init__(self, *args, important=False, **kwargs):
        self.important = important
        super().__init__(*args, **kwargs)


class ModelChunkableMixin(models.Model):
    """Mixin for Django models that makes them chunkable using the embedding_fields property"""

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

    def get_chunks(self, *, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
        splittable_content = []
        important_content = []
        embedding_fields = self._meta.model._get_embedding_fields()

        for field_ in embedding_fields:
            value = field_.get_value(self)
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
