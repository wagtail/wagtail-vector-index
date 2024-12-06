from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.models import Distance

from wagtail_vector_index.storage.base import (
    StorageProvider,
    StorageVectorIndexMixinProtocol,
)
from wagtail_vector_index.storage.models import Document, DocumentQuerySet


@dataclass
class ProviderConfig:
    HOST: str
    API_KEY: str | None = None


if TYPE_CHECKING:
    MixinBase = StorageVectorIndexMixinProtocol["QdrantStorageProvider"]
else:
    MixinBase = object


class QdrantIndexMixin(MixinBase):
    def __init__(self, **kwargs: Any) -> None:
        self.index_name = self.__class__.__name__
        self.storage_provider = self._get_storage_provider()
        super().__init__(**kwargs)

    def rebuild_index(self) -> None:
        self.storage_provider.client.delete_collection(collection_name=self.index_name)
        self.storage_provider.client.create_collection(
            collection_name=self.index_name,
            vectors_config=qdrant_models.VectorParams(
                size=512, distance=Distance.COSINE
            ),
        )
        self.upsert(documents=self.get_documents())

    def upsert(self, *, documents: Iterable[Document]) -> None:
        points = [
            qdrant_models.PointStruct(
                id=document.pk,
                vector=document.vector,
                payload=document.metadata,
            )
            for document in documents
        ]
        self.storage_provider.client.upsert(
            collection_name=self.index_name, points=points
        )

    def delete(self, *, document_ids: Sequence[str]) -> None:
        self.storage_provider.client.delete(
            collection_name=self.index_name,
            points_selector=qdrant_models.PointIdsList(points=document_ids),
        )

    def get_similar_documents(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> DocumentQuerySet:
        """
        Retrieve similar documents from Qdrant.

        Args:
            query_vector (Sequence[float]): The query vector to find similar documents for.
            limit (int): The maximum number of similar documents to return.
            similarity_threshold (float): The minimum similarity score for returned documents.
                                          Range is [0, 1] where 1 is most similar.
                                          0 means no threshold (default).

        Returns:
            Generator[Document, None, None]: A generator of similar documents.

        Note:
            Qdrant uses cosine similarity by default, where higher scores indicate
            more similar vectors. The similarity_threshold is used directly as
            Qdrant's score_threshold.
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        # Convert similarity threshold to score threshold
        # For Qdrant with cosine similarity, we can use the similarity_threshold directly
        score_threshold = similarity_threshold if similarity_threshold > 0 else None

        similar_documents = self.storage_provider.client.search(
            collection_name=self.index_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        similar_object_keys = [
            doc["payload"]["object_keys"][0] for doc in similar_documents
        ]

        return Document.objects.for_keys(similar_object_keys)


class QdrantStorageProvider(StorageProvider[ProviderConfig, QdrantIndexMixin]):
    config_class = ProviderConfig
    index_mixin = QdrantIndexMixin

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.client = QdrantClient(url=self.config.HOST, api_key=self.config.API_KEY)

    def rebuild_indexes(self) -> None:
        pass
