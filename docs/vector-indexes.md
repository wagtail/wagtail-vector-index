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
from wagtail_vector_index.storage.django import VectorIndexedMixin, EmbeddingField


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
from wagtail_vector_index.storage.django import (
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

NOTE: Where `vector_index_class` is set on a subclass of `VectorIndexedMixin` this way, the index will be registered with `wagtail-vector-index` automatically. In more complex cases, you may need to register your index class manually (read on for an example).

### Indexing across models

One of the things you might want to do with a custom index is query across multiple models, or on a subset of models. If the models share a common concrete parent model (e.g. Wagtail's `Page` model), then this can acheived by included them together in a custom vector index.

To do this, override `querysets` or `_get_querysets()` on your custom Vector Index class:

```python
from wagtail_vector_index.storage.django import (
    EmbeddableFieldsVectorIndexMixin,
    DefaultStorageVectorIndex,
)


class MyVectorIndex(EmbeddableFieldsVectorIndexMixin, DefaultStorageVectorIndex):
    querysets = [
        InformationPage.objects.all(),
        BlogPage.objects.filter(name__startswith="AI: "),
    ]
```

Indexes of this nature must be registered with `wagtail-vector-index` before they can be used. The best place to do this is in the `ready()` method of an `AppConfig` class within your project. You may find it helpful to save your custom index and any other related code to a new `vector_index` app in your project; in which case, `vector_index/apps.py` might look something like this:

```python
# vector_index/apps.py

from django.apps import AppConfig


class VectorIndexConfig(AppConfig):
    default_auto_field = "django.db.models.AutoField"
    name = "yourprojectname.vector_index"

    def ready(self):
        from wagtail_vector_index.index import registry

        from .indexes import MyEmbeddableFieldsVectorIndex

        registry.register()(MyEmbeddableFieldsVectorIndex)
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
from wagtail_vector_index.storage.django import (
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

A simple vector index implements three public methods: `search`, `query`, and `find_similar`.

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

## Using Similarity Threshold in Vector Search

### Understanding Similarity in Vector Search

Vector search operates on the principle of finding documents or items that are "similar" to a given query or reference item. This similarity is typically measured by the distance between vectors in a high-dimensional space. Before we dive into similarity thresholds, it's crucial to understand this concept.

### What is a Similarity Threshold?

A similarity threshold in vector search helps control the relevance of results. It filters out less relevant documents, improving result quality and query performance.

The similarity threshold represents a float value between 0 and 1, indicating the degree of similarity.

- 0 means no threshold (the system returns all results within the specified limit.)
- 1 means maximum similarity (the system returns only exact matches)
- Values in between filter results based on their similarity to the query

The similarity threshold has a significant impact on the number of returned results. Higher threshold values lead to fewer results, but these are potentially more relevant, highlighting the trade-off between result quantity and relevance.

### Implementing Similarity Threshold in Wagtail Vector Index

In Wagtail Vector Index, the `VectorIndex` class includes a `similarity_threshold` parameter in key methods:

* `query`: This method is used for querying the vector index.
* `find_similar`: This method finds similar objects in the vector index.
* `search`: This method performs a search in the vector index.

Each of these methods includes a `similarity_threshold` parameter, allowing you to control the similarity threshold for that specific operation.

### Best Practices and Considerations

1. **Choosing a Threshold**: Start with a lower threshold (e.g., 0.5) and adjust based on your specific use case and the quality of results.
2. Performance Impact: Optimistically, higher thresholds can significantly improve query performance by reducing the number of results processed. This potential for optimization is a key advantage of vector search.
3. **Result Set Size**: Be aware that high thresholds might significantly reduce the number of results. Always check if your result set is empty, and consider lowering the threshold if necessary.
4. **Backend Differences**: While we strive for consistency, different vector search backends (e.g., pgvector, Qdrant, Weaviate) may calculate similarity slightly differently. Test thoroughly with your specific backend.
5. **Combining with Limit**: The `similarity_threshold` parameter works in conjunction with the `limit` parameter. Results are first filtered by the similarity threshold and then limited to the specified number.

## Practical Applications

Consider these scenarios where adjusting the similarity threshold can be beneficial:

1. **Content Recommendations**: In a content recommendation system, starting with a lower threshold to cast a wide net and then gradually increasing it as you gather more user data underscores the value of your work in refining recommendations.
2. **Semantic Search**: A higher threshold in a semantic search engine can ensure that it returns more relevant results, thereby improving the user experience.
3. **Duplicate Detection: A very high threshold is appropriate to catch only the closest matches when looking for near-duplicate content.

### Debugging and Tuning

If you're not getting the expected results:

1. Try lowering the threshold to see if more relevant results appear.
2. Check the similarity scores of your results (if available) to understand the distribution.
3. Consider the nature of your data and queries. Some domains require lower thresholds to capture relevant semantic relationships.

Remember, the optimal threshold can vary depending on your specific use case, data, and embedding model. Experimentation and iterative tuning are often necessary to find the best balance between precision and recall for your application.
