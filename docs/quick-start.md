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
from wagtail_vector_index.models import VectorIndexedMixin, EmbeddingField


class MyPage(VectorIndexedMixin, Page):
    body = models.TextField()

    embedding_fields = [EmbeddingField("title"), EmbeddingField("body")]
```

A `ModelVectorIndex` will be generated for your model which can be accessed using the `get_vector_index()` classmethod, e.g.:

```python
index = MyPage.get_vector_index()
```

If you want more control over how content is indexed, you can instead create your own indexes. See [./customising.md](Customising) for more details.

Now you can index your content, see [./using-indexes](Using Indexes) for how to make these indexes work for you.

## Updating indexes

To update all indexes, run the `update_vector_indexes` management command:

```
python manage.py update_vector_indexes
```

To skip the prompt, use the `--noinput` flag.

## Using event-stream (WagtailVectorIndexSSEConsumer)

`WagtailVectorIndexSSEConsumer` is an asynchronous HTTP consumer designed for handling Server-Sent Events (SSE) for streaming responses from queries using the vector index in real-time. Using the consumer requires ASGI ([uvicorn](https://pypi.org/project/uvicorn/), [Daphne](https://pypi.org/project/daphne/) etc.) along with [django-channels](https://pypi.org/project/django-channels/).

You can configure channels using the [official guide](https://channels.readthedocs.io/en/3.x/installation.html). At a minimum, install the `channels` package and add it to `INSTALLED_APPS` in your settings file, and configure support for ASGI.

```python
# settings.py

INSTALLED_APPS = [
    "channels",
    # ...
]
```

Next, you will need to define a new consumer inheriting from `WagtailVectorIndexSSEConsumer`, and assign a Wagtail page model for the vector index you'd like to use.

!!! Note
    The `AuthMiddleware` is required to provide user context to the consumer.


```python
# app_name/asgi.py
import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path, re_path
from wagtail_vector_index.consumers import WagtailVectorIndexSSEConsumer


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings.production")
django_asgi_app = get_asgi_application()


application = ProtocolTypeRouter(
    {
        "http": URLRouter(
            [
                path(
                    "chat-query-sse/",
                    AuthMiddlewareStack(WagtailVectorIndexSSEConsumer.as_asgi()),
                ),
                re_path(r"", get_asgi_application()),
            ]
        ),
    }
)
```

You should now be able to query the consumer using the [EventSource](https://developer.mozilla.org/en-US/docs/Web/API/EventSource) API. The snippet below is an example implementation:

```javascript
function chatQuery(query, pageType) {
    const es = new EventSource(
        `/chat-query-sse/?query=${query}&page_type=${pageType}`,
    );

    es.onmessage = (e) => {
        console.log(e.data);
        // Do something
    };
    es.onerror = () => {
        // Ending an EventSource object from the server results in an error.
        // Close the EventSource here to prevent repeated requests.
        es.close();
    };
}
```

### Known issues

Asynchronous support in Django is fairly new and `WagtailVectorIndexSSEConsumer` can't tell when a client disconnects from an event-stream. This may result in queries being processed by the server as zombie threads.
