# Installation

1. `python -m pip install wagtail-vector-index`
2. Add `wagtail_vector_index` to your `INSTALLED_APPS`
3. Add an AI backend configuration (any backend supported by [https://github.com/tomusher/every-ai](EveryAI)):
    ```python
    WAGTAIL_VECTOR_INDEX_AI_BACKENDS = {
        "default": {
            "BACKEND": "openai",
            "CONFIG": {
                "api_key": "foo"
            }
        }
    }
    ```
4. Set up a Vector [./backends.md](backend) if necessary
