import unittest

import pytest
from factories import ExamplePageFactory
from faker import Faker
from testapp.models import ExamplePage
from wagtail_vector_index.storage import (
    registry,
)
from wagtail_vector_index.storage.base import VectorIndex
from wagtail_vector_index.storage.models import EmbeddingField

fake = Faker()


def test_registry():
    expected_class_names = [
        "ExamplePageIndex",
        "ExampleModelIndex",
        "DifferentPageIndex",
        "MultiplePageVectorIndex",
    ]
    assert set(registry._registry.keys()) == set(expected_class_names)


def test_indexed_model_has_vector_index():
    index = ExamplePage.vector_index
    assert index.__class__.__name__ == "ExamplePageIndex"


def test_register_custom_vector_index():
    custom_index = type("MyVectorIndex", (VectorIndex,), {})()
    registry.register_index(custom_index)
    assert registry["MyVectorIndex"] == custom_index


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
def test_index_get_documents_returns_at_least_one_document_per_page():
    pages = ExamplePageFactory.create_batch(10)
    index = registry["ExamplePageIndex"]
    index.rebuild_index()
    documents = index.get_documents()
    found_pages = {document.metadata.get("object_id") for document in documents}

    assert found_pages == {str(page.pk) for page in pages}


@pytest.mark.django_db
def test_similar_returns_no_duplicates(mocker):
    pages = ExamplePageFactory.create_batch(10)
    vector_index = ExamplePage.vector_index

    def gen_pages(cls, *args, **kwargs):
        yield from pages

    mocker.patch.object(
        vector_index.get_converter(),
        "bulk_from_documents",
        autospec=True,
        side_effect=gen_pages,
    )

    case = unittest.TestCase()

    # We expect 9 results without the page itself.
    actual = vector_index.find_similar(pages[0], limit=100, include_self=False)
    case.assertCountEqual(actual, pages[1:])

    # We expect 10 results with the page itself.
    actual = vector_index.find_similar(pages[0], limit=100, include_self=True)
    case.assertCountEqual(actual, pages)


DEDUPLICATE_LIST_TESTDATA = [
    pytest.param([3, 1, 1, 2], None, [3, 1, 2]),
    pytest.param([3, 1, 1, 2], [], [3, 1, 2]),
    pytest.param([67, 333, 50, 10, 2, 2, 3, 333], [2], [67, 333, 50, 10, 3]),
    pytest.param([67, 333, 50, 10, 2, 2, 3, 333], [2, 3], [67, 333, 50, 10]),
]


@pytest.mark.parametrize("input_list,exclusions,expected", DEDUPLICATE_LIST_TESTDATA)
def test_deduplicate_list(input_list, exclusions, expected):
    vector_index = ExamplePage.vector_index

    assert vector_index._deduplicate_list(input_list, exclusions=exclusions)
