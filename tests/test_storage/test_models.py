from wagtail_vector_index.storage.models import (
    DefaultStorageVectorIndex,
    build_vector_index_base_for_storage_provider,
)


def test_build_vector_index_base_for_default_storage_provider(settings):
    from wagtail_vector_index.storage.numpy import NumpyIndexMixin

    settings.WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
        "default": {
            "STORAGE_PROVIDER": "wagtail_vector_index.storage.numpy.NumpyStorageProvider",
        }
    }

    VectorIndexBase = build_vector_index_base_for_storage_provider("default")
    assert VectorIndexBase.__name__ == "DefaultVectorIndex"
    assert issubclass(VectorIndexBase, NumpyIndexMixin)


def test_build_vector_index_base_for_alias_storage_provider(settings):
    from wagtail_vector_index.storage.pgvector import PgvectorIndexMixin

    settings.WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
        "default": {
            "STORAGE_PROVIDER": "wagtail_vector_index.storage.numpy.NumpyStorageProvider",
        },
        "pgvector": {
            "STORAGE_PROVIDER": "wagtail_vector_index.storage.pgvector.PgvectorStorageProvider",
        },
    }

    VectorIndexBase = build_vector_index_base_for_storage_provider("pgvector")
    assert VectorIndexBase.__name__ == "PgvectorVectorIndex"
    assert issubclass(VectorIndexBase, PgvectorIndexMixin)


def test_default_storage_vector_index(settings):
    from wagtail_vector_index.storage.numpy import NumpyIndexMixin

    settings.WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
        "default": {
            "STORAGE_PROVIDER": "wagtail_vector_index.storage.numpy.NumpyStorageProvider",
        }
    }

    assert issubclass(DefaultStorageVectorIndex, NumpyIndexMixin)
