# AI Backends

Wagtail Vector Index can be configured to use different backends to support different AI services.

There are two types of backends available for customisation:

-   Embedding Backends - used to generate vector representations of text
-   Chat Backends - used to generate 'chat' responses from queries

Backends and their associated settings can be customised using the `WAGTAIL_VECTOR_INDEX` setting in your Django project settings file:

```python
WAGTAIL_VECTOR_INDEX = {
    "CHAT_BACKENDS": {
        "default": {
            "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-3.5-turbo",
            },
        },
    },
    "EMBEDDING_BACKENDS": {
        "default": {
            "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "ada-002",
            },
        }
    },
}
```

The following backends are currently available:

- [LiteLLM](./litellm.md) (default)
- [LLM](./llm.md)
