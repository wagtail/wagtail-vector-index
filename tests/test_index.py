import unittest

import pytest
from factories import (
    DifferentPageFactory,
    ExampleModelFactory,
    ExamplePageFactory,
)
from faker import Faker
from testapp.models import DifferentPage, ExampleModel, ExamplePage
from wagtail_vector_index.storage import registry
from wagtail_vector_index.storage.base import VectorIndex
from wagtail_vector_index.storage.django import EmbeddingField, ModelKey
from wagtail_vector_index.storage.models import Document

fake = Faker()


@pytest.fixture
def test_objects():
    return [
        ExampleModelFactory.create(title="Very similar to test"),
        ExampleModelFactory.create(title="Somewhat similar to test"),
        ExampleModelFactory.create(title="Not similar at all"),
    ]


@pytest.fixture
def document_generator(test_objects, get_vector_for_text):
    def gen_documents(cls, *args, **kwargs):
        for obj in test_objects:
            vector = get_vector_for_text(obj.title)
            yield Document(
                object_keys=[ModelKey.from_instance(obj)],
                metadata={
                    "title": obj.title,
                    "object_id": str(obj.pk),
                },
                vector=vector,
            )

    return gen_documents


@pytest.fixture
def async_document_generator(test_objects, get_vector_for_text):
    async def gen_documents(cls, *args, **kwargs):
        for obj in test_objects:
            vector = get_vector_for_text(obj.title)
            yield Document(
                object_keys=[ModelKey.from_instance(obj)],
                metadata={"title": obj.title, "object_id": str(obj.pk)},
                vector=vector,
            )

    return gen_documents


@pytest.fixture
def mock_vector_index(
    mocker, mock_embedding_backend, document_generator, async_document_generator
):
    vector_index = ExamplePage.vector_index

    mocker.patch.object(
        vector_index, "get_embedding_backend", return_value=mock_embedding_backend
    )

    mocker.patch(
        "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.bulk_to_documents",
        side_effect=document_generator,
    )

    mocker.patch(
        "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.abulk_to_documents",
        side_effect=async_document_generator,
    )

    return vector_index


class TestRegistry:
    def test_registry(self):
        expected_class_names = [
            "ExamplePageIndex",
            "ExampleModelIndex",
            "DifferentPageIndex",
            "MultiplePageVectorIndex",
        ]
        assert set(registry._registry.keys()) == set(expected_class_names)

    def test_indexed_model_has_vector_index(self):
        index = ExamplePage.vector_index
        assert index.__class__.__name__ == "ExamplePageIndex"

    def test_register_custom_vector_index(self):
        custom_index = type("MyVectorIndex", (VectorIndex,), {})()
        registry.register_index(custom_index)
        assert registry["MyVectorIndex"] == custom_index


class TestEmbeddingFields:
    def test_get_embedding_fields_count(self, patch_embedding_fields):
        with patch_embedding_fields(
            ExamplePage, [EmbeddingField("test"), EmbeddingField("another_test")]
        ):
            assert len(ExamplePage._get_embedding_fields()) == 2

    def test_embedding_fields_override(self, patch_embedding_fields):
        with patch_embedding_fields(
            ExamplePage, [EmbeddingField("test"), EmbeddingField("test")]
        ):
            assert len(ExamplePage._get_embedding_fields()) == 1

    def test_checking_search_fields_errors_with_invalid_field(
        self, patch_embedding_fields
    ):
        with patch_embedding_fields(ExamplePage, [EmbeddingField("foo")]):
            errors = ExamplePage.check()
            assert "wagtailai.WA001" in [error.id for error in errors]


class TestIndexOperations:
    @pytest.mark.django_db
    def test_index_get_documents_returns_at_least_one_document_per_page(self):
        pages = ExampleModelFactory.create_batch(10)
        index = registry["ExampleModelIndex"]
        index.rebuild_index()
        documents = index.get_documents()
        found_pages = {
            ModelKey(document.object_keys[0]).object_id for document in documents
        }

        assert found_pages == {str(page.pk) for page in pages}

    @pytest.mark.django_db
    def test_index_with_multiple_models(self):
        example_pages = ExamplePageFactory.create_batch(5)
        different_pages = DifferentPageFactory.create_batch(5)
        index = registry["MultiplePageVectorIndex"]
        index.rebuild_index()

        example_pages_ids = {str(page.pk) for page in example_pages}
        different_page_ids = {str(page.pk) for page in different_pages}
        found_page_ids = {
            ModelKey(document.object_keys[0]).object_id
            for document in index.get_documents()
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


class TestSimilarityOperations:
    @pytest.mark.django_db
    def test_similar_returns_no_duplicates(self, mocker):
        pages = ExampleModelFactory.create_batch(10)
        vector_index = ExamplePage.vector_index

        def gen_pages(cls, *args, **kwargs):
            yield from pages

        mocker.patch(
            "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.bulk_from_documents",
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
    def test_find_similar_with_similarity_threshold(self, mocker):
        pages = ExampleModelFactory.create_batch(10)
        vector_index = ExamplePage.vector_index

        def gen_pages(cls, *args, **kwargs):
            yield from pages

        mocker.patch(
            "wagtail_vector_index.storage.django.EmbeddableFieldsDocumentConverter.bulk_from_documents",
            side_effect=gen_pages,
        )

        # We expect 9 results without the page itself.
        actual = vector_index.find_similar(
            pages[0], limit=100, include_self=False, similarity_threshold=0.5
        )
        assert set(actual) == set(pages[1:]), f"Expected {pages[1:]}, but got {actual}"

        # We expect 10 results with the page itself.
        actual = vector_index.find_similar(
            pages[0], limit=100, include_self=True, similarity_threshold=0.5
        )
        assert set(actual) == set(pages), f"Expected {pages}, but got {actual}"

    @pytest.mark.django_db(transaction=True)
    async def test_afind_similar(self, mock_vector_index):
        objs = [obj async for obj in ExampleModel.objects.all()]
        actual = await mock_vector_index.afind_similar(
            objs[0], limit=100, include_self=False
        )
        assert set(actual) == set(objs[1:]), f"Expected {objs[1:]}, but got {actual}"


class TestQueryOperations:
    @pytest.mark.django_db
    def test_query_passes_sources_to_backend(self, mocker):
        ExampleModelFactory.create_batch(2)
        index = ExamplePage.vector_index
        documents = index.get_documents()[:2]

        def get_similar_documents(query_embedding, limit=0, similarity_threshold=0.0):
            yield from documents

        query_mock = mocker.patch("conftest.ChatMockBackend.chat")
        expected_content = "\n".join([doc.content for doc in documents])
        similar_documents_mock = mocker.patch.object(index, "get_similar_documents")
        similar_documents_mock.side_effect = get_similar_documents
        index.query("")
        first_call_messages = query_mock.call_args.kwargs["messages"]
        assert first_call_messages[1] == {"content": expected_content, "role": "system"}

    @pytest.mark.django_db
    def test_query_with_similarity_threshold(self, mocker):
        ExampleModelFactory.create_batch(2)
        index = ExamplePage.vector_index
        documents = index.get_documents()[:2]

        def get_similar_documents(query_embedding, limit=0, similarity_threshold=0.5):
            yield from documents

        query_mock = mocker.patch("conftest.ChatMockBackend.chat")
        expected_content = "\n".join([doc.content for doc in documents])
        similar_documents_mock = mocker.patch.object(index, "get_similar_documents")
        similar_documents_mock.side_effect = get_similar_documents
        index.query("", similarity_threshold=0.5)
        first_call_messages = query_mock.call_args.kwargs["messages"]
        assert first_call_messages[1] == {"content": expected_content, "role": "system"}

    @pytest.mark.django_db
    @pytest.mark.parametrize(
        "similarity_threshold, expected_count, expected_titles",
        [
            (0.9, 0, set()),
            (0.6, 1, {"Very similar to test"}),
            (0.1, 2, {"Very similar to test", "Somewhat similar to test"}),
            (
                None,
                3,
                {
                    "Very similar to test",
                    "Somewhat similar to test",
                    "Not similar at all",
                },
            ),
        ],
    )
    def test_search_with_similarity_threshold(
        self, mock_vector_index, similarity_threshold, expected_count, expected_titles
    ):
        kwargs = {"limit": 100}
        if similarity_threshold is not None:
            kwargs["similarity_threshold"] = similarity_threshold

        results = list(mock_vector_index.search("test", **kwargs))

        assert (
            len(results) == expected_count
        ), f"Expected {expected_count} results, got {len(results)}"

        if expected_count > 0:
            assert {result.title for result in results} == expected_titles
