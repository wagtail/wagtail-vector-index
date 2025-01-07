import factory
import pytest
from asgiref.sync import async_to_sync
from factories import (
    BookPageFactory,
    DocumentFactory,
    FilmPageFactory,
    VideoGameFactory,
)
from faker import Faker
from testapp.models import BookPage, FilmPage
from wagtail_vector_index.ai import get_embedding_backend
from wagtail_vector_index.storage.chunking import EmbeddingField
from wagtail_vector_index.storage.django import (
    ModelFromDocumentOperator,
    ModelKey,
    ModelLabel,
    ModelToDocumentOperator,
    PreparedObject,
    PreparedObjectCollection,
)
from wagtail_vector_index.storage.models import Document

fake = Faker()


class TestChunking:
    @pytest.mark.django_db
    def test_get_chunks_splits_content_into_multiple_chunks(
        self, patch_embedding_fields
    ):
        with patch_embedding_fields(BookPage, [EmbeddingField("body")]):
            body = fake.text(max_nb_chars=1000)
            instance = BookPageFactory.build(title="Important Title", body=body)
            chunks = instance.get_chunks(chunk_size=100)
            assert len(chunks) > 1

    @pytest.mark.django_db
    def test_get_chunks_adds_important_field_to_each_chunk(
        self, patch_embedding_fields
    ):
        with patch_embedding_fields(
            BookPage,
            [EmbeddingField("title", important=True), EmbeddingField("body")],
        ):
            body = fake.text(max_nb_chars=200)
            instance = BookPageFactory.build(title="Important Title", body=body)
            chunks = instance.get_chunks(chunk_size=100)
            assert all(chunk.startswith(instance.title) for chunk in chunks)


class TestFromDocument:
    def test_extract_model_class_from_label(self):
        label = ModelLabel("testapp.BookPage")
        model_class = ModelFromDocumentOperator._model_class_from_label(label)
        assert model_class == BookPage

    @pytest.mark.django_db
    def test_get_keys_by_model_label(self):
        book_pages = BookPageFactory.create_batch(3)
        film_pages = FilmPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.BookPage:{page.pk}"] for page in book_pages]
                + [[f"testapp.FilmPage:{page.pk}"] for page in film_pages]
            ),
        )
        keys_by_model_label = ModelFromDocumentOperator._get_keys_by_model_label(
            documents
        )

        assert len(keys_by_model_label) == 2
        assert "testapp.BookPage" in keys_by_model_label
        assert "testapp.FilmPage" in keys_by_model_label
        assert len(keys_by_model_label["testapp.BookPage"]) == 3
        assert len(keys_by_model_label["testapp.FilmPage"]) == 3

    @pytest.mark.django_db
    def test_get_models_by_key(self):
        book_pages = BookPageFactory.create_batch(3)
        film_pages = FilmPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.BookPage:{page.pk}"] for page in book_pages]
                + [[f"testapp.FilmPage:{page.pk}"] for page in film_pages]
            ),
        )
        keys_by_model_label = ModelFromDocumentOperator._get_keys_by_model_label(
            documents
        )
        models_by_key = ModelFromDocumentOperator._get_models_by_key(
            keys_by_model_label
        )
        assert len(models_by_key) == 6
        assert all(
            isinstance(model, (BookPage, FilmPage)) for model in models_by_key.values()
        )

    @pytest.mark.django_db
    def test_from_document_returns_model_object(self):
        instance = BookPageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        document = DocumentFactory.create(
            object_keys=[f"testapp.BookPage:{instance.pk}"],
        )
        operator = ModelFromDocumentOperator()
        recovered_instance = operator.from_document(document)
        assert isinstance(recovered_instance, BookPage)
        assert recovered_instance.pk == instance.pk

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects(self):
        instances = BookPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            3,
            object_keys=factory.Iterator(
                [[f"testapp.BookPage:{page.pk}"] for page in instances]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 3
        assert all(isinstance(instance, BookPage) for instance in recovered_instances)
        assert all(
            instance.pk in [page.pk for page in instances]
            for instance in recovered_instances
        )

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects_in_order(self):
        instances = BookPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            3,
            object_keys=factory.Iterator(
                [[f"testapp.BookPage:{page.pk}"] for page in instances]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert recovered_instances == instances

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects_for_multiple_models(self):
        book_pages = BookPageFactory.create_batch(3)
        film_pages = FilmPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.BookPage:{page.pk}"] for page in book_pages]
                + [[f"testapp.FilmPage:{page.pk}"] for page in film_pages]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 6
        assert all(
            isinstance(instance, (BookPage, FilmPage))
            for instance in recovered_instances
        )
        assert all(
            instance.pk in [page.pk for page in book_pages + film_pages]
            for instance in recovered_instances
        )

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_deduplicated_model_objects(self):
        instance = BookPageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        documents = DocumentFactory.create_batch(
            3,
            object_keys=[f"testapp.BookPage:{instance.pk}"],
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 1
        assert recovered_instances[0].pk == instance.pk


class TestToDocument:
    @pytest.mark.django_db
    def test_generate_documents_returns_documents(self):
        instance = BookPageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        operator = ModelToDocumentOperator()
        documents = list(
            operator.to_documents(
                [instance], embedding_backend=get_embedding_backend("default")
            )
        )
        assert len(documents) == 1
        assert documents[0].content == f"{instance.title}\n{instance.body}"

    @pytest.mark.django_db
    def test_bulk_generate_documents_returns_documents(self):
        instances = BookPageFactory.create_batch(3)
        operator = ModelToDocumentOperator()
        documents = list(
            operator.to_documents(
                instances, embedding_backend=get_embedding_backend("default")
            )
        )
        assert len(documents) == 3
        assert all(
            document.content == f"{instance.title}\n{instance.body}"
            for document, instance in zip(documents, instances, strict=False)
        )

    @pytest.mark.django_db
    def test_bulk_generate_documents_returns_documents_for_multiple_models(self):
        book_pages = BookPageFactory.create_batch(3)
        film_pages = FilmPageFactory.create_batch(3)
        operator = ModelToDocumentOperator()
        documents = list(
            operator.to_documents(
                book_pages + film_pages,
                embedding_backend=get_embedding_backend("default"),
            )
        )
        assert len(documents) == 6
        assert all(
            document.content
            == f"{instance.title}\n{instance.body or instance.description}"
            for document, instance in zip(
                documents, book_pages + film_pages, strict=False
            )
        )

    @pytest.mark.django_db
    def test_to_documents_batches_objects(self, mocker):
        instances = BookPageFactory.create_batch(10)
        operator = ModelToDocumentOperator()
        to_documents_batch_mock = mocker.patch.object(operator, "_to_documents_batch")
        list(
            operator.to_documents(
                instances,
                embedding_backend=get_embedding_backend("default"),
                batch_size=2,
            )
        )
        assert to_documents_batch_mock.call_count == 5


class TestConverter:
    @pytest.mark.django_db
    def test_returns_original_object(self, patch_embedding_fields):
        with patch_embedding_fields(BookPage, [EmbeddingField("body")]):
            instance = BookPageFactory.create(
                title="Important Title", body=fake.text(max_nb_chars=200)
            )
            converter = ModelToDocumentOperator()
            document = next(
                converter.to_documents(
                    [instance], embedding_backend=get_embedding_backend("default")
                )
            )
            recovered_instance = ModelFromDocumentOperator().from_document(document)
            assert isinstance(recovered_instance, BookPage)
            assert recovered_instance.pk == instance.pk


@pytest.mark.django_db
def test_convert_single_document_to_object():
    instance = BookPageFactory.create(
        title="Important Title", body=fake.text(max_nb_chars=200)
    )
    documents = list(
        ModelToDocumentOperator().to_documents(
            [instance], embedding_backend=get_embedding_backend("default")
        )
    )
    recovered_instance = ModelFromDocumentOperator().from_document(documents[0])
    assert isinstance(recovered_instance, BookPage)
    assert recovered_instance.pk == instance.pk


@pytest.mark.django_db
def test_convert_multiple_documents_to_objects():
    book_pages = BookPageFactory.create_batch(5)
    film_pages = FilmPageFactory.create_batch(5)
    video_games = VideoGameFactory.create_batch(5)
    all_objects = list(book_pages + film_pages + video_games)
    documents = list(
        ModelToDocumentOperator().to_documents(
            all_objects, embedding_backend=get_embedding_backend("default")
        )
    )
    recovered_objects = list(ModelFromDocumentOperator().bulk_from_documents(documents))
    assert recovered_objects == all_objects


class TestToDocumentOperatorAsync:
    @pytest.mark.django_db(transaction=True)
    def test_ato_documents_batch(self, mock_embedding_backend):
        instance = VideoGameFactory.create(
            title="Important Title", description=fake.text(max_nb_chars=200)
        )
        operator = ModelToDocumentOperator()

        documents = async_to_sync(operator._ato_documents_batch)(
            [instance], embedding_backend=mock_embedding_backend
        )

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(instance.title in doc.content for doc in documents)

    @pytest.mark.django_db(transaction=True)
    def test_aupdate_object_collection_with_new_documents(self, mock_embedding_backend):
        instance = VideoGameFactory.create()
        collection = PreparedObjectCollection.prepare_objects(
            objects=[instance],
            embedding_backend=mock_embedding_backend,
        )

        operator = ModelToDocumentOperator()
        async_to_sync(operator._aupdate_object_collection_with_new_documents)(
            collection, mock_embedding_backend
        )

        assert any(obj.new_documents for obj in collection)
        assert all(
            isinstance(doc, Document) for obj in collection for doc in obj.new_documents
        )


class TestPreparedObject:
    @pytest.mark.django_db
    def test_needs_updating_when_no_existing_documents(self):
        instance = BookPageFactory.build()
        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1", "chunk2"],
        )
        assert prepared_object.needs_updating is True

    @pytest.mark.django_db
    def test_needs_updating_when_chunks_match(self):
        instance = BookPageFactory.build()
        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1", "chunk2"],
            existing_documents=[
                DocumentFactory.build(content="chunk1"),
                DocumentFactory.build(content="chunk2"),
            ],
        )
        assert prepared_object.needs_updating is False

    @pytest.mark.django_db
    def test_needs_updating_when_chunks_differ(self):
        instance = BookPageFactory.build()
        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1", "chunk2"],
            existing_documents=[
                DocumentFactory.build(content="chunk1"),
                DocumentFactory.build(content="different chunk"),
            ],
        )
        assert prepared_object.needs_updating is True

    @pytest.mark.django_db
    def test_documents_returns_new_documents_when_present(self):
        instance = BookPageFactory.build()
        new_docs = [DocumentFactory.build(), DocumentFactory.build()]
        existing_docs = [DocumentFactory.build(), DocumentFactory.build()]

        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1"],
            new_documents=new_docs,
            existing_documents=existing_docs,
        )
        assert prepared_object.documents == new_docs

    @pytest.mark.django_db
    def test_documents_returns_existing_documents_when_no_new_ones(self):
        instance = BookPageFactory.build()
        existing_docs = [DocumentFactory.build(), DocumentFactory.build()]

        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1"],
            existing_documents=existing_docs,
        )
        assert prepared_object.documents == existing_docs


class TestPreparedObjectCollection:
    @pytest.mark.django_db
    def test_prepare_objects(self, patch_embedding_fields):
        with patch_embedding_fields(BookPage, [EmbeddingField("body")]):
            instances = BookPageFactory.create_batch(3)

            collection = PreparedObjectCollection.prepare_objects(
                objects=instances,
                embedding_backend=get_embedding_backend("default"),
            )

            assert len(collection.objects) == 3
            assert all(isinstance(obj, PreparedObject) for obj in collection)
            assert all(obj.chunks for obj in collection)

    @pytest.mark.django_db
    def test_get_chunk_mapping(self):
        instances = BookPageFactory.build_batch(2)
        prepared_objects = [
            PreparedObject(
                key=ModelKey.from_instance(instance),
                object=instance,
                chunks=["chunk1", "chunk2"],
            )
            for instance in instances
        ]
        collection = PreparedObjectCollection(objects=prepared_objects)

        chunk_mapping = collection.get_chunk_mapping()
        assert len(chunk_mapping) == 4  # 2 instances * 2 chunks each
        assert all(isinstance(key, ModelKey) for key in chunk_mapping)

    @pytest.mark.django_db
    def test_prepare_new_documents(self):
        instances = BookPageFactory.create_batch(2)
        prepared_objects = [
            PreparedObject(
                key=ModelKey.from_instance(instance),
                object=instance,
                chunks=["chunk1", "chunk2"],
            )
            for instance in instances
        ]
        collection = PreparedObjectCollection(objects=prepared_objects)

        # Mock embedding vectors
        embedding_vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        collection.prepare_new_documents(embedding_vectors)

        assert all(len(obj.new_documents) == 2 for obj in collection)
        assert all(
            all(isinstance(doc, Document) for doc in obj.new_documents)
            for obj in collection
        )
