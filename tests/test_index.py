import unittest

import pytest
from factories import ExamplePageFactory
from faker import Faker
from testapp.models import ExamplePage
from wagtail_vector_index.ai import get_embedding_backend
from wagtail_vector_index.index import (
    VectorIndex,
    get_vector_indexes,
    registry,
)
from wagtail_vector_index.models import EmbeddingField

fake = Faker()


def test_get_vector_indexes():
    indexes = get_vector_indexes()
    expected_class_names = [
        "ExamplePageIndex",
        "ExampleModelIndex",
        "DifferentPageIndex",
        "MultiplePageVectorIndex",
    ]
    index_class_names = [index.__class__.__name__ for index in indexes.values()]
    assert set(index_class_names) == set(expected_class_names)


def test_indexed_model_has_vector_index():
    index = ExamplePage.get_vector_index()
    assert index.__class__.__name__ == "ExamplePageIndex"


def test_register_custom_vector_index():
    custom_index = type("MyVectorIndex", (VectorIndex,), {})
    registry.register()(custom_index)
    index_classes = [index.__class__ for index in get_vector_indexes().values()]
    assert custom_index in index_classes


def test_get_embedding_fields_count(patch_embedding_fields):
    with patch_embedding_fields(
        ExamplePage, [EmbeddingField("test"), EmbeddingField("another_test")]
    ):
        assert len(ExamplePage._get_embedding_fields()) == 2


def test_embedding_fields_override(patch_embedding_fields):
    # In the same vein as Wagtail's search index fields, if there are
    # multiple fields of the same type with the same name, only one
    # should be returned
    with patch_embedding_fields(
        ExamplePage, [EmbeddingField("test"), EmbeddingField("test")]
    ):
        assert len(ExamplePage._get_embedding_fields()) == 1


def test_checking_search_fields_errors_with_invalid_field(patch_embedding_fields):
    with patch_embedding_fields(ExamplePage, [EmbeddingField("foo")]):
        errors = ExamplePage.check()
        assert "wagtailai.WA001" in [error.id for error in errors]


@pytest.mark.django_db
def test_get_split_content_adds_important_field_to_each_split(patch_embedding_fields):
    with patch_embedding_fields(
        ExamplePage, [EmbeddingField("title", important=True), EmbeddingField("body")]
    ):
        body = fake.text(max_nb_chars=200)
        instance = ExamplePageFactory.create(title="Important Title", body=body)
        splits = instance._get_split_content(
            embedding_backend=get_embedding_backend("default")
        )
        assert all(split.startswith(instance.title) for split in splits)


@pytest.mark.django_db
def test_index_get_documents_returns_at_least_one_document_per_page():
    pages = ExamplePageFactory.create_batch(10)
    index = get_vector_indexes()["ExamplePageIndex"]
    index.rebuild_index()
    documents = index.get_documents()
    found_pages = {document.metadata.get("object_id") for document in documents}

    assert found_pages == {str(page.pk) for page in pages}


@pytest.mark.django_db
def test_similar_returns_no_duplicates(mocker):
    pages = ExamplePageFactory.create_batch(10)
    vector_index = ExamplePage.get_vector_index()

    def gen_pages(cls: ExamplePage, *args, **kwargs):
        yield from pages

    mocker.patch.object(
        vector_index.object_type,
        "bulk_from_documents",
        autospec=True,
        side_effect=gen_pages,
    )

    case = unittest.TestCase()

    # We expect 9 results without the page itself.
    actual = vector_index.similar(pages[0], limit=100, include_self=False)
    case.assertCountEqual(actual, pages[1:])

    # We expect 10 results with the page itself.
    actual = vector_index.similar(pages[0], limit=100, include_self=True)
    case.assertCountEqual(actual, pages)
