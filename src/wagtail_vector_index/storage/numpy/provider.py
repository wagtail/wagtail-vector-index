import logging
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wagtail_vector_index.storage.base import (
    Document,
    StorageProvider,
    StorageVectorIndexMixinProtocol,
)

logger = logging.Logger(__name__)


@dataclass
class ProviderConfig: ...


if TYPE_CHECKING:
    MixinBase = StorageVectorIndexMixinProtocol["NumpyStorageProvider"]
else:
    MixinBase = object


class NumpyIndexMixin(MixinBase):
    def rebuild_index(self) -> None:
        self.get_documents()

    def upsert(self, *, documents: Iterable[Document]) -> None:
        pass

    def delete(self, *, document_ids: Sequence[str]) -> None:
        pass

    def similarity_search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> Generator[Document, None, None]:
        similarities = []
        for document in self.get_documents():
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


class NumpyStorageProvider(StorageProvider[ProviderConfig, NumpyIndexMixin]):
    config_class = ProviderConfig
    index_mixin = NumpyIndexMixin
