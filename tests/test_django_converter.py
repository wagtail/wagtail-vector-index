import factory
import pytest
from asgiref.sync import sync_to_async
from factories import (
    AsyncExampleModelFactory,
    DifferentPageFactory,
    DocumentFactory,
    ExampleModelFactory,
    ExamplePageFactory,
)
from faker import Faker
from testapp.models import DifferentPage, ExamplePage
from wagtail_vector_index.ai import get_embedding_backend
from wagtail_vector_index.storage.django import (
    EmbeddableFieldsDocumentConverter,
    EmbeddableFieldsObjectChunkerOperator,
    EmbeddingField,
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
        with patch_embedding_fields(ExamplePage, [EmbeddingField("body")]):
            body = fake.text(max_nb_chars=1000)
            instance = ExamplePageFactory.build(title="Important Title", body=body)
            chunker = EmbeddableFieldsObjectChunkerOperator()
            chunks = chunker.chunk_object(instance, chunk_size=100)
            assert len(chunks) > 1

    @pytest.mark.django_db
    def test_get_chunks_adds_important_field_to_each_chunk(
        self, patch_embedding_fields
    ):
        with patch_embedding_fields(
            ExamplePage,
            [EmbeddingField("title", important=True), EmbeddingField("body")],
        ):
            body = fake.text(max_nb_chars=200)
            instance = ExamplePageFactory.build(title="Important Title", body=body)
            chunker = EmbeddableFieldsObjectChunkerOperator()
            chunks = chunker.chunk_object(instance, chunk_size=100)
            assert all(chunk.startswith(instance.title) for chunk in chunks)


class TestFromDocument:
    def test_extract_model_class_from_label(self):
        label = ModelLabel("testapp.ExamplePage")
        model_class = ModelFromDocumentOperator._model_class_from_label(label)
        assert model_class == ExamplePage

    @pytest.mark.django_db
    def test_get_keys_by_model_label(self):
        example_pages = ExamplePageFactory.create_batch(3)
        different_pages = DifferentPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.ExamplePage:{page.pk}"] for page in example_pages]
                + [[f"testapp.DifferentPage:{page.pk}"] for page in different_pages]
            ),
        )
        keys_by_model_label = ModelFromDocumentOperator._get_keys_by_model_label(
            documents
        )

        assert len(keys_by_model_label) == 2
        assert "testapp.ExamplePage" in keys_by_model_label
        assert "testapp.DifferentPage" in keys_by_model_label
        assert len(keys_by_model_label["testapp.ExamplePage"]) == 3
        assert len(keys_by_model_label["testapp.DifferentPage"]) == 3

    @pytest.mark.django_db
    def test_get_models_by_key(self):
        example_pages = ExamplePageFactory.create_batch(3)
        different_pages = DifferentPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.ExamplePage:{page.pk}"] for page in example_pages]
                + [[f"testapp.DifferentPage:{page.pk}"] for page in different_pages]
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
            isinstance(model, (ExamplePage, DifferentPage))
            for model in models_by_key.values()
        )

    @pytest.mark.django_db
    def test_from_document_returns_model_object(self):
        instance = ExamplePageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        document = DocumentFactory.create(
            object_keys=[f"testapp.ExamplePage:{instance.pk}"],
        )
        operator = ModelFromDocumentOperator()
        recovered_instance = operator.from_document(document)
        assert isinstance(recovered_instance, ExamplePage)
        assert recovered_instance.pk == instance.pk

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects(self):
        instances = ExamplePageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            3,
            object_keys=factory.Iterator(
                [[f"testapp.ExamplePage:{page.pk}"] for page in instances]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 3
        assert all(
            isinstance(instance, ExamplePage) for instance in recovered_instances
        )
        assert all(
            instance.pk in [page.pk for page in instances]
            for instance in recovered_instances
        )

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects_in_order(self):
        instances = ExamplePageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            3,
            object_keys=factory.Iterator(
                [[f"testapp.ExamplePage:{page.pk}"] for page in instances]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert recovered_instances == instances

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_model_objects_for_multiple_models(self):
        example_pages = ExamplePageFactory.create_batch(3)
        different_pages = DifferentPageFactory.create_batch(3)
        documents = DocumentFactory.create_batch(
            6,
            object_keys=factory.Iterator(
                [[f"testapp.ExamplePage:{page.pk}"] for page in example_pages]
                + [[f"testapp.DifferentPage:{page.pk}"] for page in different_pages]
            ),
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 6
        assert all(
            isinstance(instance, (ExamplePage, DifferentPage))
            for instance in recovered_instances
        )
        assert all(
            instance.pk in [page.pk for page in example_pages + different_pages]
            for instance in recovered_instances
        )

    @pytest.mark.django_db
    def test_bulk_from_documents_returns_deduplicated_model_objects(self):
        instance = ExamplePageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        documents = DocumentFactory.create_batch(
            3,
            object_keys=[f"testapp.ExamplePage:{instance.pk}"],
        )
        operator = ModelFromDocumentOperator()
        recovered_instances = list(operator.bulk_from_documents(documents))
        assert len(recovered_instances) == 1
        assert recovered_instances[0].pk == instance.pk


class TestToDocument:
    @pytest.mark.django_db
    def test_generate_documents_returns_documents(self):
        instance = ExamplePageFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)
        documents = list(
            operator.to_documents(
                [instance], embedding_backend=get_embedding_backend("default")
            )
        )
        assert len(documents) == 1
        assert documents[0].content == f"{instance.title}\n{instance.body}"

    @pytest.mark.django_db
    def test_bulk_generate_documents_returns_documents(self):
        instances = ExamplePageFactory.create_batch(3)
        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)
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
        example_pages = ExamplePageFactory.create_batch(3)
        different_pages = DifferentPageFactory.create_batch(3)
        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)
        documents = list(
            operator.to_documents(
                example_pages + different_pages,
                embedding_backend=get_embedding_backend("default"),
            )
        )
        assert len(documents) == 6
        assert all(
            document.content == f"{instance.title}\n{instance.body}"
            for document, instance in zip(
                documents, example_pages + different_pages, strict=False
            )
        )

    @pytest.mark.django_db
    def test_to_documents_batches_objects(self, mocker):
        instances = ExamplePageFactory.create_batch(10)
        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)
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
        with patch_embedding_fields(ExamplePage, [EmbeddingField("body")]):
            instance = ExamplePageFactory.create(
                title="Important Title", body=fake.text(max_nb_chars=200)
            )
            converter = EmbeddableFieldsDocumentConverter()
            document = next(
                converter.to_documents(
                    [instance], embedding_backend=get_embedding_backend("default")
                )
            )
            recovered_instance = converter.from_document(document)
            assert isinstance(recovered_instance, ExamplePage)
            assert recovered_instance.pk == instance.pk


@pytest.mark.django_db
def test_convert_single_document_to_object():
    converter = EmbeddableFieldsDocumentConverter()
    instance = ExamplePageFactory.create(
        title="Important Title", body=fake.text(max_nb_chars=200)
    )
    documents = list(
        converter.to_documents(
            [instance], embedding_backend=get_embedding_backend("default")
        )
    )
    recovered_instance = converter.from_document(documents[0])
    assert isinstance(recovered_instance, ExamplePage)
    assert recovered_instance.pk == instance.pk


@pytest.mark.django_db
def test_convert_multiple_documents_to_objects():
    converter = EmbeddableFieldsDocumentConverter()
    example_objects = ExampleModelFactory.create_batch(5)
    example_pages = ExamplePageFactory.create_batch(5)
    different_pages = DifferentPageFactory.create_batch(5)
    all_objects = list(example_objects + example_pages + different_pages)
    documents = list(
        converter.to_documents(
            all_objects, embedding_backend=get_embedding_backend("default")
        )
    )
    recovered_objects = list(converter.bulk_from_documents(documents))
    assert recovered_objects == all_objects


class TestToDocumentOperatorAsync:
    @pytest.mark.django_db(transaction=True)
    async def test_ato_documents_batch(self, mock_embedding_backend):
        instance = await AsyncExampleModelFactory.create(
            title="Important Title", body=fake.text(max_nb_chars=200)
        )
        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)

        documents = await operator._ato_documents_batch(
            [instance], embedding_backend=mock_embedding_backend
        )

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(instance.title in doc.content for doc in documents)

    @pytest.mark.django_db(transaction=True)
    async def test_aupdate_object_collection_with_new_documents(
        self, mock_embedding_backend
    ):
        instance = await AsyncExampleModelFactory.create()
        collection = await sync_to_async(PreparedObjectCollection.prepare_objects)(
            objects=[instance],
            chunker_operator=EmbeddableFieldsObjectChunkerOperator(),
            embedding_backend=mock_embedding_backend,
        )

        operator = ModelToDocumentOperator(EmbeddableFieldsObjectChunkerOperator)
        await operator._aupdate_object_collection_with_new_documents(
            collection, mock_embedding_backend
        )

        assert any(obj.new_documents for obj in collection)
        assert all(
            isinstance(doc, Document) for obj in collection for doc in obj.new_documents
        )


class TestPreparedObject:
    @pytest.mark.django_db
    def test_needs_updating_when_no_existing_documents(self):
        instance = ExamplePageFactory.build()
        prepared_object = PreparedObject(
            key=ModelKey.from_instance(instance),
            object=instance,
            chunks=["chunk1", "chunk2"],
        )
        assert prepared_object.needs_updating is True

    @pytest.mark.django_db
    def test_needs_updating_when_chunks_match(self):
        instance = ExamplePageFactory.build()
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
        instance = ExamplePageFactory.build()
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
        instance = ExamplePageFactory.build()
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
        instance = ExamplePageFactory.build()
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
        with patch_embedding_fields(ExamplePage, [EmbeddingField("body")]):
            instances = ExamplePageFactory.create_batch(3)
            chunker = EmbeddableFieldsObjectChunkerOperator()

            collection = PreparedObjectCollection.prepare_objects(
                objects=instances,
                chunker_operator=chunker,
                embedding_backend=get_embedding_backend("default"),
            )

            assert len(collection.objects) == 3
            assert all(isinstance(obj, PreparedObject) for obj in collection)
            assert all(obj.chunks for obj in collection)

    @pytest.mark.django_db
    def test_get_chunk_mapping(self):
        instances = ExamplePageFactory.build_batch(2)
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
        instances = ExamplePageFactory.create_batch(2)
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
