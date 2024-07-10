import unittest

import pytest
from factories import DifferentPageFactory, ExamplePageFactory
from faker import Faker
from testapp.models import DifferentPage, ExamplePage
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
def test_index_with_multiple_models():
    example_pages = ExamplePageFactory.create_batch(5)
    different_pages = DifferentPageFactory.create_batch(5)
    index = registry["MultiplePageVectorIndex"]
    index.rebuild_index()

    example_pages_ids = {str(page.pk) for page in example_pages}
    different_page_ids = {str(page.pk) for page in different_pages}
    found_page_ids = {
        document.metadata["object_id"] for document in index.get_documents()
    }

    assert found_page_ids == example_pages_ids.union(different_page_ids)

    similar_result = list(index.find_similar(DifferentPage.objects.first()))
    assert len(similar_result) > 1
    for p in similar_result:
        assert isinstance(p, (ExamplePage, DifferentPage))

    search_result = list(index.search("test"))
    assert len(search_result) > 1
    for p in search_result:
        assert isinstance(p, (ExamplePage, DifferentPage))


@pytest.mark.django_db
def test_similar_returns_no_duplicates(mocker):
    pages = ExamplePageFactory.create_batch(10)
    vector_index = ExamplePage.vector_index

    def gen_pages(cls, *args, **kwargs):
        yield from pages

    mocker.patch(
        "wagtail_vector_index.storage.models.EmbeddableFieldsDocumentConverter.bulk_from_documents",
        side_effect=gen_pages,
    )

    case = unittest.TestCase()

    # We expect 9 results without the page itself.
    actual = vector_index.find_similar(pages[0], limit=100, include_self=False)
    case.assertCountEqual(actual, pages[1:])

    # We expect 10 results with the page itself.
    actual = vector_index.find_similar(pages[0], limit=100, include_self=True)
    case.assertCountEqual(actual, pages)


@pytest.mark.django_db
def test_query_passes_sources_to_backend(mocker):
    ExamplePageFactory.create_batch(2)
    index = ExamplePage.vector_index
    documents = index.get_documents()[:2]

    def get_similar_documents(query_embedding, limit=0):
        yield from documents

    query_mock = mocker.patch("conftest.ChatMockBackend.chat")
    expected_content = "\n".join([doc.metadata["content"] for doc in documents])
    similar_documents_mock = mocker.patch.object(index, "get_similar_documents")
    similar_documents_mock.side_effect = get_similar_documents
    index.query("")
    first_call_messages = query_mock.call_args.kwargs["messages"]
    assert first_call_messages[1] == {"content": expected_content, "role": "system"}
