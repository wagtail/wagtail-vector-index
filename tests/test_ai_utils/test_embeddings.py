import pytest
from wagtail_vector_index.ai_utils import embeddings


def test_get_default_embedding_dimmensions_output_for_known_model():
    assert embeddings.get_default_embedding_output_dimensions("ada-002") == 1536


def test_get_default_token_limit_for_unknown_model():
    with pytest.raises(
        embeddings.EmbeddingOutputDimensionsNotFound, match="echo-123-02"
    ):
        embeddings.get_default_embedding_output_dimensions("echo-123-02")
