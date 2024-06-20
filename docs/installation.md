# Installation

1. Install Wagtail Vector Index. You need to specify optional dependencies for vector
   backend and AI backends (for embeddings and chat models) and it will depend on your
   project's circumstances.
    * Select an [AI provider backend](./ai-backends/index.md), we recommend starting with `LiteLLM`.
    * For the vector storage provider, you can choose between:
        - pgvector _(Postgres database extension)_:
          `python -m pip install wagtail-vector-index[pgvector]`
        - Qdrant: `python -m pip install wagtail-vector-index[qdrant]`
        - Weaviate: `python -m pip install wagtail-vector-index[weaviate]`
        - NumPy: `python -m pip install wagtail-vector-index[numpy]` *(not recommended
         for big databases or production applications due to scale issues)*
        - Read more about storage providers on [the specific documentation page: **Storage Providers**](./storage-providers.md).
    * In your final installation call, you should comma-separate the optional
      dependencies you want to install, e.g.
      `python -m pip install wagtail-vector-index[litellm,pgvector]`.
2. Add `wagtail_vector_index` to your `INSTALLED_APPS` in your Django project
settings file.
   ```python
   INSTALLED_APPS = [
       # ...
       "wagtail_vector_index",
       # ...
   ]
   ```
    - If you are using the pgvector backend, you also need to add the specific
      pgvector app:
      ```python
      INSTALLED_APPS = [
          # ...
          "wagtail_vector_index",
          "wagtail_vector_index.storage.pgvector",
          # ...
      ]
      ```
3. Add an AI backend configuration to your Django project settings file. Wagtail
   Vector Index ships with a backend for the
   [LiteLLM package](https://www.litellm.ai) which you configure for OpenAI in your project
   settings:
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
                    "MODEL_ID": "text-embedding-ada-002",
                },
            }
        },
    }
    ```
   * You can supply your OpenAI key in the `OPENAI_API_KEY` environment variable
     which is read directly by the openai library (installed by the llm package).
4. Run database migrations.
   ```sh
   ./manage.py migrate
   ```
