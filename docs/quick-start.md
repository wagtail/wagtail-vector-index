# Getting Started

Wagtail Vector Index combines integrations with AI 'embedding' APIs and vector databases to provide tools to perform advanced AI-powered querying across content.

To do this:

-   You set up models/pages to be searchable.
-   Wagtail Vector Index splits the content of those pages into chunks and fetches embeddings from the configured AI backend.
-   It then stores all those embeddings in the configured vector database.
-   When querying, the query is converted to an embedding and, using the vector database, is compared to the embeddings for all your existing content.

## What's an Embedding?

An embedding is a big list (vector) of floating point numbers that represent your content in some way. Models like OpenAI's `ada-002` can take content and turn it in to a list of numbers such that content that is similar will have a similar list of numbers.

This way, when you provide a query, we can use the same model to get an embedding of that query and do some maths (cosine similarity) to see what content in your vector database is similar to your query.

## Indexing Your Models/Pages

To index your models:

1. Add Wagtail Vector Index's `VectorIndexedMixin` mixin to your model
2. Set `embedding_fields` to a list of `EmbeddingField`s representing the fields you want to be included in the embeddings

```python
from django.db import models
from wagtail.models import Page
from wagtail_vector_index.index.models import VectorIndexedMixin, EmbeddingField


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]
```

An `EmbeddableFieldsVectorIndex` will be generated for your model which can be accessed using the `vector_index` class property, e.g.:

```python
index = MyPage.vector_index
```

If you want more control over how content is indexed, you can instead create your own indexes. See [./vector-indexes.md](Vector Indexes) for more details.

Now you can index your content, see [./using-indexes](Using Indexes) for how to make these indexes work for you.

## Updating indexes

To update all indexes, run the `update_vector_indexes` management command:

```
python manage.py update_vector_indexes
```

To skip the prompt, use the `--noinput` flag.
