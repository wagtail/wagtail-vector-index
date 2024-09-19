import pytest
from factories import BookPageFactory, FilmPageFactory, VideoGameFactory
from faker import Faker
from testapp.models import BookPage
from wagtail_vector_index.ai import get_embedding_backend
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsDocumentConverter,
    EmbeddingField,
)

fake = Faker()


@pytest.mark.django_db
def test_get_split_content_adds_important_field_to_each_split(patch_embedding_fields):
    with patch_embedding_fields(
        BookPage, [EmbeddingField("title", important=True), EmbeddingField("body")]
    ):
        body = fake.text(max_nb_chars=200)
        instance = BookPageFactory.create(title="Important Title", body=body)
        converter = EmbeddableFieldsDocumentConverter()
        splits = converter._get_split_content(instance, chunk_size=100)
        assert all(split.startswith(instance.title) for split in splits)


@pytest.mark.django_db
def test_convert_single_document_to_object():
    converter = EmbeddableFieldsDocumentConverter()
    instance = BookPageFactory.create(
        title="Important Title", body=fake.text(max_nb_chars=200)
    )
    documents = list(
        converter.to_documents(
            instance, embedding_backend=get_embedding_backend("default")
        )
    )
    recovered_instance = converter.from_document(documents[0])
    assert isinstance(recovered_instance, BookPage)
    assert recovered_instance.pk == instance.pk


@pytest.mark.django_db
def test_convert_multiple_documents_to_objects():
    converter = EmbeddableFieldsDocumentConverter()
    example_objects = VideoGameFactory.create_batch(5)
    example_pages = BookPageFactory.create_batch(5)
    different_pages = FilmPageFactory.create_batch(5)
    all_objects = list(example_objects + example_pages + different_pages)
    documents = converter.bulk_to_documents(
        all_objects, embedding_backend=get_embedding_backend("default")
    )
    recovered_objects = list(converter.bulk_from_documents(documents))
    assert recovered_objects == all_objects
