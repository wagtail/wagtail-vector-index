import pytest
from factories import ExamplePageFactory
from faker import Faker
from testapp.models import ExamplePage
from wagtail_vector_index.index import registry

fake = Faker()

pytestmark = pytest.mark.django_db


class IndexOperations:
    """Common assertions for index operations"""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        ExamplePageFactory.create_batch(10)
        ExamplePage.vector_index.rebuild_index()

    def get_index(self):
        raise NotImplementedError("Must be implemented in subclass")

    def test_search(self):
        index = self.get_index()
        results = index.search("")
        assert len(results) == 5

    def test_query(self):
        index = self.get_index()
        results = index.query("")
        assert len(results.sources) == 5


class TestIndexOperationsFromModel(IndexOperations):
    def get_index(self):
        return ExamplePage.vector_index


class TestIndexOperationsFromRegistry(IndexOperations):
    def get_index(self):
        return registry["ExamplePageIndex"]()
