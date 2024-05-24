# LiteLLM

LiteLLM is the default backend provider, making use of the [LiteLLM library](https://github.com/BerriAI/litellm) to communicate with [many LLM providers](https://docs.litellm.ai/docs/providers), as well as local providers and custom APIs.

## Chat Backend

The LiteLLM chat backend is enabled by default, using OpenAI's `gpt-3.5-turbo` model. Adding an `OPENAI_API_KEY` to your environment is enough to get started using this backend, however the behaviour of this backend can be customised in your Django settings.

Tu use a specific model, specify the `MODEL_ID` as below. The names of supported models can be found in the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

```python
WAGTAIL_VECTOR_INDEX = {
    "CHAT_BACKENDS": {
        "default": {
            "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-4o",
            },
        },
    },
}
```

The behaviour of the chat backend can be customised by passing [input parameters](https://docs.litellm.ai/docs/completion/input) to LiteLLM. To provide a certain input parameter by default, specify `DEFAULT_PARAMETERS` in the backend config.

For example; to customise the temperature of all responses:

```python
WAGTAIL_VECTOR_INDEX = {
    "CHAT_BACKENDS": {
        "default": {
            "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMChatBackend",
            "CONFIG": {
                "MODEL_ID": "gpt-4o",
                "DEFAULT_PARAMETERS": {"temperature": 0.8},
            },
        },
    },
}
```

## Embedding Backend

The LiteLLM embedding backend is enabled by default using OpenAI's `ada-002` embedding model. Adding an `OPENAI_API_KEY` to your environment is enough to get started using this backend, however the behaviour of this backend can be customised in your Django settings.

Tu use a specific model, specify the `MODEL_ID` as below. The names of supported models can be found in the [LiteLLM documentation](https://docs.litellm.ai/docs/embedding/supported_embedding).

```python
WAGTAIL_VECTOR_INDEX = {
    "EMBEDDING_BACKENDS": {
        "default": {
            "CLASS": "wagtail_vector_index.ai_utils.backends.litellm.LiteLLMEmbeddingBackend",
            "CONFIG": {
                "MODEL_ID": "vertex_ai/textembedding-gecko",
            },
        },
    },
}
```

## Using local models

To use local models with the LiteLLM backend, we recommend using the [Ollama provider](https://docs.litellm.ai/docs/providers/ollama).
