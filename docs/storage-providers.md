# Storage Providers

Wagtail Vector Index supports multiple storage providers for storage and querying of content embeddings.

The vector backend can be configured using the `WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS` setting:

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "STORAGE_PROVIDER": "wagtail_vector_index.storage.pgvector.PgvectorStorageProvider",
    }
}
```

## NumPy Provider

!!! warning

    The numpy based provider is not recommended for production use and large databases
    because it is not designed to scale. It is a good starting point for local
    experimentation due to not requiring any additional services to be set up.

The Numpy provider is the default provider, but can be configured explicitly with:

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "STORAGE_PROVIDER": "wagtail_vector_index.storage.numpy.NumpyBackend",
    }
}
```

It does not need any additional processes to work, but it does require the
`numpy` package to be installed. We recommend you do that by installing
Wagtail Vector Index with an optional dependency:

```sh
python -m pip install wagtail-vector-index[numpy]
```

This provide siterates through each embedding in the database, running similarity
checks against content in-memory. This may be useful for small sets of content,
but will likely be slow and consume large amounts of memory as the size of
indexed content increases. For this reason it is not recommended for production
use.


## pgvector Provider

This provider makes use of the [pgvector](https://github.com/pgvector/pgvector) extension
in the Postgres database. This may be a good option if you already use a Postgres database
and do not want to add more services to your infrastructure.

!!! warning

    This provider won't work with non-PostgreSQL databases but you might see
    no errors (it will fail silently). It's advised to use Postgres across
    all your environments (including local development environment) if you
    use the pgvector provider.

You must ensure your Postgres instance supports the
[pgvector](https://github.com/pgvector/pgvector) extension. To use this
backend, you should install Wagtail Vector Index with an optional dependency:

```sh
python -m pip install wagtail-vector-index[pgvector]
```

To configure the vector index, use the following settings in your Django
project settings:

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "STORAGE_PROVIDER": "wagtail_vector_index.backends.pgvector.backend.PgvectorBackend",
    }
}
```

You also need to install another Django application because the pgvector backend
contains its own database model.

```python
INSTALLED_APPS = [
    # ...
    "wagtail_vector_index",
    "wagtail_vector_index.storage.pgvector",
    # ...
]
```

And you need to run the migrations:

```sh
./manage.py migrate
```

## Qdrant Provider

The [Qdrant](https://qdrant.tech/) provider supports both the cloud and self-hosted
versions of the Qdrant vector database.

To use this provider, you should install Wagtail Vector Index with an optional
dependency:

```sh
python -m pip install wagtail-vector-index[qdrant]
```

You need to configure the provider in your Django project
settings file:

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "STORAGE_PROVIDER": "wagtail_vector_index.storage.qdrant.QdrantBackend",
        "HOST": "https://location/of/qdrant/cluster",
        "API_KEY": "your_qdrant_cloud_api_key",  # Not required for self-hosted installations
    }
}
```

## Weaviate Provider

The [Weaviate](https://weaviate.io/) provider supports both the cloud and
self-hosted versions of the Weaviate vector database.

To use this provider, you should install Wagtail Vector Index with an optional
dependency:

```sh
python -m pip install wagtail-vector-index[weaviate]
```

You need to configure the vector index provider in your Django project
settings file:

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "BACKEND": "wagtail_vector_index.storage.weaviate.WeaviateBackend",
        "HOST": "https://location/of/weaviate/cluster",
        "API_KEY": "your_weaviate_api_key",
    }
}
```
