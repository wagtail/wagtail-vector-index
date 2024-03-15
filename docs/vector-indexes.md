
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

You'll then be able to call the `vector_index` property on your model to get the generated `ModelVectorIndex`.

```python
index = MyPage.vector_index
```

The `VectorIndexedMixin` class is made up of two other mixins:

- `EmbeddableFieldsMixin` which lets you define `embedding_fields` on a model
Vector Indexes are a feature of Wagtail Vector Index that allows you to query site content using AI tools. They provide a way to turn Django models, Wagtail pages, and anything else in to embeddings which are stored in your database (and optionally in another Vector Database), and then query that content for similarity or using natural language.
- `GeneratedIndexMixin` which provides a method that automatically generates a Vector Index for you, and adds the `vector_index` convenience property for accessing that index.
