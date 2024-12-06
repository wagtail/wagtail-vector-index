import pytest
from factories import DifferentPageFactory, ExamplePageFactory
from testapp.models import ExamplePage, MultiplePageVectorIndex
from wagtail_vector_index.storage.filters import ObjectTypeFilter, QuerySetFilter
from wagtail_vector_index.storage.models import Document, DocumentQuerySet


@pytest.fixture
def test_pages(db):
    test_page1 = ExamplePageFactory.create(
        title="Example Page 1", body="This is the first test page"
    )
    test_page2 = ExamplePageFactory.create(
        title="Example Page 2", body="This is the second test page"
    )
    test_other_page = DifferentPageFactory.create(
        title="Other Page", body="This is another type of page"
    )

    return [test_page1, test_page2, test_other_page]


@pytest.fixture
def document_queryset(test_pages):
    MultiplePageVectorIndex().get_documents()
    return Document.objects.all()


def test_queryset_filter(document_queryset):
    queryset = ExamplePage.objects.filter(title="Example Page 1")
    filter = QuerySetFilter(queryset)

    filtered_documents = filter.apply(document_queryset)

    assert len(filtered_documents) == 1
    assert filtered_documents[0].object_keys[0].startswith("testapp.ExamplePage")
    assert "Example Page 1" in filtered_documents[0].content


def test_object_type_filter(document_queryset, test_pages):
    filter = ObjectTypeFilter(ExamplePage)

    filtered_documents = filter.apply(document_queryset)

    assert len(filtered_documents) == 2
    assert all(
        doc.object_keys[0].startswith("testapp.ExamplePage")
        for doc in filtered_documents
    )


def test_combined_filters(document_queryset, test_pages):
    queryset_filter = QuerySetFilter(
        ExamplePage.objects.filter(title__contains="Example")
    )
    object_type_filter = ObjectTypeFilter(ExamplePage)

    filtered_documents = object_type_filter.apply(
        queryset_filter.apply(document_queryset)
    )

    assert len(filtered_documents) == 2
    assert all(
        doc.object_keys[0].startswith("testapp.ExamplePage")
        for doc in filtered_documents
    )
    assert all("Example" in doc.content for doc in filtered_documents)


class CustomFilter:
    def apply(self, documents: DocumentQuerySet) -> DocumentQuerySet:
        return documents.filter(content__icontains="first")


def test_custom_filter(document_queryset):
    custom_filter = CustomFilter()

    filtered_documents = custom_filter.apply(document_queryset)

    assert len(filtered_documents) == 1
    assert "first" in filtered_documents[0].content.lower()


def test_filter_chaining(document_queryset):
    queryset_filter = QuerySetFilter(ExamplePage.objects.all())
    object_type_filter = ObjectTypeFilter(ExamplePage)
    custom_filter = CustomFilter()

    filtered_documents = custom_filter.apply(
        object_type_filter.apply(queryset_filter.apply(document_queryset))
    )

    assert len(filtered_documents) == 1
    assert filtered_documents[0].object_keys[0].startswith("testapp.ExamplePage")
    assert "first" in filtered_documents[0].content.lower()
