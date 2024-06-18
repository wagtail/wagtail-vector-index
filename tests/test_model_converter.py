import pytest
from factories import ExamplePageFactory
from faker import Faker
from testapp.models import ExamplePage
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsDocumentConverter,
    EmbeddingField,
)

fake = Faker()


@pytest.mark.django_db
def test_get_split_content_adds_important_field_to_each_split(patch_embedding_fields):
    with patch_embedding_fields(
        ExamplePage, [EmbeddingField("title", important=True), EmbeddingField("body")]
    ):
        body = fake.text(max_nb_chars=200)
        instance = ExamplePageFactory.create(title="Important Title", body=body)
        converter = EmbeddableFieldsDocumentConverter(ExamplePage)
        splits = converter._get_split_content(instance, chunk_size=100)
        assert all(split.startswith(instance.title) for split in splits)
