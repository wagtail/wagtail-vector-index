class EmbeddingOutputDimensionsNotFound(Exception):
    pass


def get_default_embedding_output_dimensions(model_id: str) -> int:
    match model_id:
        case "ada-002":
            return 1536
        case _:
            raise EmbeddingOutputDimensionsNotFound(model_id)
