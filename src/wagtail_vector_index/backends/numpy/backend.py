import logging
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from wagtail_vector_index.backends.base import Backend, Index
from wagtail_vector_index.index.base import Document

logger = logging.Logger(__name__)


@dataclass
class BackendConfig: ...


class NumpyIndex(Index):
    def upsert(self, *, documents: Iterable[Document]) -> None:
        pass

    def delete(self, *, document_ids: Sequence[str]) -> None:
        pass

    def similarity_search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> Generator[Document, None, None]:
        similarities = []
        vector_index = self.get_vector_index()
        for document in vector_index.get_documents():
            cosine_similarity = (
                np.dot(query_vector, document.vector)
                / np.linalg.norm(query_vector)
                * np.linalg.norm(document.vector)
            )
            similarities.append((cosine_similarity, document))

        sorted_similarities = sorted(
            similarities, key=lambda pair: pair[0], reverse=True
        )
        for document in [pair[1] for pair in sorted_similarities][:limit]:
            yield document


class NumpyBackend(Backend[BackendConfig, NumpyIndex]):
    config_class = BackendConfig

    def get_index(self, index_name: str) -> NumpyIndex:
        return NumpyIndex(index_name)

    def create_index(self, index_name: str, *, vector_size: int) -> NumpyIndex:
        return self.get_index(index_name)

    def delete_index(self, index_name: str) -> None:
        pass
