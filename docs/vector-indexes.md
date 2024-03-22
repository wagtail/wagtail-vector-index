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

### Setting a custom DocumentConverter

To convert your models/Pages in to `Documents` ready for ingestion in to the vector index, the package uses a `DocumentConverter`. By default, the `VectorIndexableMixin` uses the `EmbeddableFieldsDocumentConverter`, which knows how to parse the `embedded_fields` you specify,


## Creating your own index

If you want to customise your vector index, you can build your own subclass of `EmbeddableFieldsVectorIndex` and configure your model to use it.

```python
from wagtail_vector_index.index import EmbeddableFieldsVectorIndex


class MyEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    embedding_backend = MyEmbeddingBackend


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]

    vector_index_class = MyEmbeddableFieldsVectorIndex
```


## Indexing across models

If you want to be able to query across multiple models, or on a subset of models, they need to be in a vector index together.

To do this, you can define and register your own `EmbeddableFieldsVectorIndex`:

```python
from wagtail_vector_index.index.models import EmbeddableFieldsVectorIndex


class MyEmbeddableFieldsVectorIndex(EmbeddableFieldsVectorIndex):
    querysets = [
        MyModel.objects.all(),
        MyOtherModel.objects.filter(name__startswith="AI: "),
    ]
```

Once populated (with the `update_vector_indexes` management command), this can be queried just like an automatically generated index:

```python
index = MyEmbeddableFieldsVectorIndex()
index.query("Are you suggesting that coconuts migrate?")
```

## Customising embedding splits

Due to token limitations in AI models, content from indexed models is split up in to chunks, with embeddings generated separately.

By default this is done by merging all `embedding_fields` together and then splitting on new paragraphs, new lines, sentences and words (getting more specific as required) until it fits within a defined split size.

To customise this behaviour, either:

- Override the `_get_text_splitter` method on a `VectorIndexedMixin` model, returning a class that conforms to the `TextSplitterProtocol`.
- Override the `_get_split_content` method on a `VectorIndexedMixin` model to split your content however you want.
