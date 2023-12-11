# Vector backends

Wagtail Vector Index supports multiple vector backends for storage and querying of content embeddings.

The vector backend can be configured using the `WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS` setting:

```python
WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS = {
    "default": {
        "BACKEND": "openai",
        "API_KEY": "abc123",
        "HOST": "https://example.com",
    }
}
```

## NumPy Backend

!!! warning

    The numpy based backend is not recommended for production use and large databases
    because it is not designed to scale. It is a good starting point for local
    experimentation due to not requiring any additional services to be set up.

The Numpy backend is the default backend, but can be configured explicitly with:

```python
WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS = {
    "default": {
        "BACKEND": "wagtail_vector_index.backends.numpy.NumpyBackend",
    }
}
```

It does not need any additional processes to work, but it does require the
`numpy` package to be installed. To use this backend, you should install Wagtail
Vector Index with an optional dependency:

```sh
python -m pip install wagtail-vector-index[numpy]
```

This backend iterates through each embedding in the database, running similarity
checks against content in-memory. This may be useful for small sets of content,
but will likely be slow and consume large amounts of memory as the size of
indexed content increases. For this reason it is not recommended for production
use.


## pgvector backend

This backend makes use of the [pgvector](https://github.com/pgvector/pgvector) extension
in the Postgres database. This may be a good option if you already use a Postgres database
and do not want to add more services to your infrastructure.

```python
WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS = {
    "default": {
        "BACKEND": "wagtail_vector_index.backends.pgvector.PgvectorBackend",
    }
}
```

You must ensure your Postgres instance supports the
[pgvector](https://github.com/pgvector/pgvector) extension. To use this
backend, you should install Wagtail Vector Index with an optional dependency:

```sh
python -m pip install wagtail-vector-index[pgvector]
```

## Qdrant Backend

The [Qdrant](https://qdrant.tech/) backend supports both the cloud and self-hosted
versions of the Qdrant vector database.

```python
WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS = {
    "default": {
        "BACKEND": "wagtail_vector_index.backends.qdrant.QdrantBackend",
        "HOST": "https://location/of/qdrant/cluster",
        "API_KEY": "your_qdrant_cloud_api_key",  # Not required for self-hosted installations
    }
}
```
To use this backend, you should install Wagtail Vector Index with an optional
dependency:

```sh
python -m pip install wagtail-vector-index[qdrant]
```

## Weaviate Backend

The [Weaviate](https://weaviate.io/) backend supports both the cloud and
self-hosted versions of the Weaviate vector database.

```python
WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS = {
    "default": {
        "BACKEND": "wagtail_vector_index.backends.weaviate.WeaviateBackend",
        "HOST": "https://location/of/weaviate/cluster",
        "API_KEY": "your_weaviate_api_key",
    }
}
```

To use this backend, you should install Wagtail Vector Index with an optional
dependency:

```sh
python -m pip install wagtail-vector-index[weaviate]
```
