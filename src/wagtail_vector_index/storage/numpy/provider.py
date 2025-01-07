import logging
from collections.abc import AsyncGenerator, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from asgiref.sync import sync_to_async

from wagtail_vector_index.storage.conversion import (
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

    def _calculate_similarities(
        self,
        query_vector: Sequence[float],
        documents: Sequence["Document"],
        similarity_threshold: float,
        limit: int,
    ) -> Generator["Document", None, None]:
        similarities = []

        for document in documents:
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

    def get_similar_documents(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> Generator["Document", None, None]:
        document_keys = [document.object_keys[0] for document in self.get_documents()]
        document_objs = Document.objects.for_keys(document_keys).apply_filters(
            self._filters
        )

        yield from self._calculate_similarities(
            query_vector, list(document_objs), similarity_threshold, limit
        )

    async def aget_similar_documents(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> AsyncGenerator["Document", None]:
        document_keys = [
            document.object_keys[0] async for document in self.aget_documents()
        ]
        document_objs = await sync_to_async(list)(
            Document.objects.for_keys(document_keys).apply_filters(self._filters)
        )

        for document in self._calculate_similarities(
            query_vector, document_objs, similarity_threshold, limit
        ):
            yield document


class NumpyStorageProvider(StorageProvider[ProviderConfig, NumpyIndexMixin]):
    config_class = ProviderConfig
    index_mixin = NumpyIndexMixin
