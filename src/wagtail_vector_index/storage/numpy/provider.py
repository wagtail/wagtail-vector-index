import logging
from collections.abc import AsyncGenerator, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wagtail_vector_index.storage.base import (
    StorageProvider,
    StorageVectorIndexMixinProtocol,
)
from wagtail_vector_index.storage.models import Document

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

    def upsert(self, *, documents: Iterable["Document"]) -> None:
        pass

    def delete(self, *, document_ids: Sequence[str]) -> None:
        pass

    def get_similar_documents(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> Generator["Document", None, None]:
        similarities = []
        document_keys = [document.object_keys[0] for document in self.get_documents()]
        document_objs = Document.objects.for_keys(document_keys).apply_filters(
            self._filters
        )

        for document in document_objs:
            cosine_similarity = (
                np.dot(query_vector, document.vector)
                / np.linalg.norm(query_vector)
                * np.linalg.norm(document.vector)
            )
            if cosine_similarity >= similarity_threshold:
                similarities.append((cosine_similarity, document))

        sorted_similarities = sorted(
            similarities, key=lambda pair: pair[0], reverse=True
        )
        for document in [pair[1] for pair in sorted_similarities][:limit]:
            yield document

    async def aget_similar_documents(
        self, query_vector, *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> AsyncGenerator["Document", None]:
        documents = self.get_similar_documents(
            query_vector, limit=limit, similarity_threshold=similarity_threshold
        )
        for document in documents:
            yield document


class NumpyStorageProvider(StorageProvider[ProviderConfig, NumpyIndexMixin]):
    config_class = ProviderConfig
    index_mixin = NumpyIndexMixin
