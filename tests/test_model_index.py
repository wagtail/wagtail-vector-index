import pytest
from factories import BookPageFactory
from faker import Faker
from testapp.models import BookPage
from wagtail_vector_index.storage import registry
from wagtail_vector_index.storage.models import Embedding

fake = Faker()

pytestmark = pytest.mark.django_db


class IndexOperations:
    """Common assertions for index operations"""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        BookPageFactory.create_batch(10)
        BookPage.vector_index.rebuild_index()

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
        return BookPage.vector_index


class TestIndexOperationsFromRegistry(IndexOperations):
    def get_index(self):
        return registry["BookPageIndex"]


def test_rebuilding_model_index_creates_embeddings():
    BookPageFactory.create_batch(10)
    index = BookPage.vector_index
    index.rebuild_index()
    assert Embedding.objects.count() == 10
