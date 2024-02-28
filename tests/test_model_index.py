import pytest
from factories import ExamplePageFactory
from testapp.models import ExamplePage
from wagtail_vector_index.index import get_vector_indexes

pytestmark = pytest.mark.django_db


class IndexOperations:
    """Common assertions for index operations"""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        ExamplePageFactory.create_batch(10)
        ExamplePage.get_vector_index().rebuild_index()

    def get_index(self):
        raise NotImplementedError("Must be implemented in subclass")

    def test_search(self):
        index = self.get_index()
        results = index.search("")
        assert len(results) == 5


class TestIndexOperationsFromModel(IndexOperations):
    def get_index(self):
        return ExamplePage.get_vector_index()


class TestIndexOperationsFromRegistry(IndexOperations):
    def get_index(self):
        return get_vector_indexes()["ExamplePageIndex"]
