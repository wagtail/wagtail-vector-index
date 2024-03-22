# Vector Indexes

Vector Indexes are a feature of Wagtail Vector Index that allows you to query site content using AI tools. They provide a way to turn Django models, Wagtail pages, and anything else in to embeddings which are stored in your database (and optionally in another Vector Database), and then query that content for similarity or using natural language.

A barebones implementation of a `VectorIndex` needs to implement one method; `get_documents` - this returns an Iterable of `Document` objects, which represent an embedding along with some metadata.

There are two ways to use Vector Indexes. Either:

-   Adding the `VectorIndexedMixin` to a Django model, which will automatically generate an Index for that model
-   Creating your own subclass of one of the `VectorIndex` base classes and an associated `DocumentConverter`.

## Automatically Generating Indexes using `VectorIndexedMixin`

If you don't need to customise the way your index behaves, you can automatically generate a Vector Index based on an existing model in your application:

1. Add Wagtail AI's `VectorIndexedMixin` mixin to your model
2. Set `embedding_fields` to a list of `EmbeddingField`s representing the fields you want to be included in the embeddings

```python
from django.db import models
from wagtail.models import Page
from wagtail_vector_index.index.models import VectorIndexedMixin, EmbeddingField


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

If you want to customise your vector index, you can build your own subclass of `EmbeddableFieldsVectorIndex` and configure your model to use it with the `vector_index_class` property:

```python
from wagtail_vector_index.index import EmbeddableFieldsVectorIndex


class MyEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    embedding_backend_alias = "openai"


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]

    vector_index_class = MyEmbeddableFieldsVectorIndex
```


### Indexing across models

One of the things you might want to do with a custom index is query across multiple models, or on a subset of models. To do this, they need to be in a vector index together.

To do this, override `querysets` or `_get_querysets()` on your `EmbeddableFieldsVectorIndex` class:

```python
from wagtail_vector_index.index.models import EmbeddableFieldsVectorIndex


class MyEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    querysets = [
        MyModel.objects.all(),
        MyOtherModel.objects.filter(name__startswith="AI: "),
    ]
```

Once populated (with the `update_vector_indexes` management command), these indexes can be queried just like an automatically generated index:

```python
index = MyEmbeddableFieldsVectorIndex()
index.query("Are you suggesting that coconuts migrate?")
```

### Setting a custom DocumentConverter

The `EmbeddableFieldsVectorIndex` class knows how to split up your page/model using a Document Converter - a class which handles the conversion between your model and a Document, which is then stored in the vector store.

By default this looks at all the `embedding_fields` on your page, builds a representation of them, splits them in to chunks and then generations embeddings for each chunk.

You might want to customise this behavior. To do this you can create your own `DocumentConverter`


```python
from wagtail_vector_index.index.models import (
    EmbeddableFieldsVectorIndex,
    EmbeddableFieldsDocumentConverter,
)


class MyDocumentConverter(EmbeddableFieldsDocumentConverter):
    def _get_split_content(self, object, *, embedding_backend):
        return object.body.split("\n")


class MyEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    querysets = [
        MyModel.objects.all(),
        MyOtherModel.objects.filter(name__startswith="AI: "),
    ]

    def get_converter_class(self):
        return MyDocumentConverter
```
