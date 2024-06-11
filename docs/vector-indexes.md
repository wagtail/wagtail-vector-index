# Vector Indexes

Vector Indexes are a feature of Wagtail Vector Index that allows you to query site content using AI tools. They provide a way to turn Django models, Wagtail pages, and anything else in to embeddings which are stored in your database (and optionally in another Vector Database), and then query that content for similarity or using natural language.

A barebones implementation of a `VectorIndex` needs to implement one method; `get_documents` - this returns an Iterable of `Document` objects, which represent an embedding along with some metadata.

There are two ways to use Vector Indexes. Either:

-   Adding the `VectorIndexedMixin` to a Django model, which will automatically generate an Index for that model
-   Creating your own subclass of one of the `VectorIndex` base classes and an associated `DocumentConverter`.

## Automatically Generating Indexes using `VectorIndexedMixin`

If you don't need to customise the way your index behaves, you can automatically generate a Vector Index based on an existing model in your application:

1. Add the `VectorIndexedMixin` mixin to your model
2. Set `embedding_fields` to a list of `EmbeddingField`s representing the fields you want to be included in the embeddings

```python
from django.db import models
from wagtail.models import Page
from wagtail_vector_index.storage.models import VectorIndexedMixin, EmbeddingField


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]
```

You'll then be able to call the `vector_index` property on your model to get the generated `EmbeddableFieldsVectorIndex`.

```python
index = MyPage.vector_index
```

The `VectorIndexedMixin` class is made up of two other mixins:

- `EmbeddableFieldsMixin` which lets you define `embedding_fields` on a model
- `GeneratedIndexMixin` which provides a method that automatically generates a Vector Index for you, and adds the `vector_index` convenience property for accessing that index.

## Creating your own index

If you want to customise your vector index, you can build your own `VectorIndex` class and configure your model to use it with the `vector_index_class` property:

```python
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsVectorIndexMixin,
    DefaultStorageVectorIndex,
)


class MyVectorIndex(EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex):
    embedding_backend_alias = "openai"


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]

    vector_index_class = MyVectorIndex
```


### Indexing across models

One of the things you might want to do with a custom index is query across multiple models, or on a subset of models. To do this, they need to be in a vector index together.

To do this, override `querysets` or `_get_querysets()` on your custom Vector Index class:

```python
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsVectorIndexMixin,
    DefaultStorageVectorIndex,
)


class MyVectorIndex(EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex):
    querysets = [
        MyModel.objects.all(),
        MyOtherModel.objects.filter(name__startswith="AI: "),
    ]
```

Once populated (with the `update_vector_indexes` management command), these indexes can be queried just like an automatically generated index:

```python
index = MyVectorIndex()
index.query("Are you suggesting that coconuts migrate?")
```

### Setting a custom DocumentConverter

The `EmbeddableFieldsVectorIndexMixin` mixin knows how to split up your page/model using a Document Converter - a class which handles the conversion between your model and a Document, which is then stored in the vector store.

By default this looks at all the `embedding_fields` on your page, builds a representation of them, splits them in to chunks and then generations embeddings for each chunk.

You might want to customise this behavior. To do this you can create your own `DocumentConverter`


```python
from wagtail_vector_index.storage.models import (
    EmbeddableFieldsVectorIndexMixin,
    EmbeddableFieldsDocumentConverter,
    DefaultStorageVectorIndex,
)


class MyDocumentConverter(EmbeddableFieldsDocumentConverter):
    def _get_split_content(self, object, *, embedding_backend):
        return object.body.split("\n")


class MyEmbeddableFieldsVectorIndex(
    EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex
):
    querysets = [
        MyModel.objects.all(),
        MyOtherModel.objects.filter(name__startswith="AI: "),
    ]

    def get_converter_class(self):
        return MyDocumentConverter
```

### How Vector Indexes are structured

A simple vector index implements three public methods: `search`, `query`, and `similar`.

To make these work in practice, a vector index also needs to:

- Know what content to query/search over
- Know where to store that content

These behaviours can be added to a basic `VectorIndex` with mixins provided by the package, or with your own custom mixins.

When working with Django/Wagtail models, knowing what content to query/search over is handled by the `EmbeddableFieldsVectorIndexMixin`. This looks at all the specified querysets, and extracts content from their `embedding_fields`, converting them to `Documents`.

To store that content somewhere, the package supports multiple Storage Providers, each with their own mixin. These mixins add behaviour for storing and querying content from a specific provider. e.g. the `PgvectorIndexMixin`, when added to a `VectorIndex`, will store that content in a `pgvector` table on your PostgreSQL database.

A `VectorIndex` with all behaviour specified through mixins would look like:

```python
class MyVectorIndex(EmebeddableFieldsVectorIndexMixin, PgvectorIndexMixin, VectorIndex):
    pass
```

To use a storage provider-specific mixin, you must first configure that [./storage-providers](storage provider) in your Django settings. If you are using multiple providers, you can specify which alias to use using the `storage_provider_alias` class attribute on your `VectorIndex`.

```python
WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS = {
    "default": {
        "STORAGE_PROVIDER": "wagtail_vector_index.storage.pgvector.PgvectorStorageProvider",
    },
    "weaviate": {
        "STORAGE_PROVIDER": "wagtail_vector_index.storage.weaviate.WeaviateStorageProvider",
    },
}


class MyPgvectorVectorIndex(
    EmebeddableFieldsVectorIndexMixin, PgvectorIndexMixin, VectorIndex
):
    storage_provider_alias = "default"


class MyWeaviateVectorIndex(
    EmebeddableFieldsVectorIndexMixin, WeaviateIndexMixin, VectorIndex
):
    storage_provider_alias = "weaviate"
```
