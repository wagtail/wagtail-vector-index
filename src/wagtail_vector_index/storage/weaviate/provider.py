import json
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import weaviate
from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.storage.base import (
    Document,
    StorageProvider,
    StorageVectorIndexMixinProtocol,
    VectorIndex,
)


@dataclass
class ProviderConfig:
    HOST: str
    API_KEY: str | None = None


if TYPE_CHECKING:
    MixinBase = StorageVectorIndexMixinProtocol["WeaviateStorageProvider"]
else:
    MixinBase = object


class WeaviateIndexMixin(MixinBase):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.storage_provider = self._get_storage_provider()
        self.index_name = self.__class__.__name__

    def rebuild_index(
        self,
    ) -> None:
        self.storage_provider.client.schema.delete_class(self.index_name)
        self.storage_provider.client.schema.create_class(
            {
                "class": self.index_name,
            }
        )
        self.upsert(documents=self.get_documents())

    def upsert(self, *, documents: Iterable[Document]) -> None:
        with self.storage_provider.client.batch as batch:
            for document in documents:
                # Store metadata as a JSON string because otherwise
                # we need to explicitly request each field back in
                # the query
                batch.add_data_object(
                    {
                        "metadata": json.dumps(document.metadata),
                        "embedding_pk": document.embedding_pk,
                    },
                    self.index_name,
                    vector=document.vector,
                )

    def delete(self, *, document_ids: Sequence[str]) -> None:
        # TODO: Handle deletion
        raise NotImplementedError

    def get_similar_documents(
        self, query_vector: Sequence[float], *, limit: int = 5, similarity_threshold: float = 0.0
    ) -> Generator[Document, None, None]:
        """
        Retrieve similar documents from Weaviate.

        Args:
            query_vector (Sequence[float]): The query vector to find similar documents for.
            limit (int): The maximum number of similar documents to return. Defaults to 5.
            similarity_threshold (float): The minimum similarity for returned documents.
                                        Range is [0, 1] where 1 is most similar.
                                        0 means no threshold (default).

        Returns:
            Generator[Document, None, None]: A generator of similar documents.

        Note:
            Weaviate uses cosine distance internally, where lower values indicate
            more similar vectors. The similarity_threshold is converted to a 
            distance threshold where distance = 1 - similarity.
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        # Convert similarity threshold to distance threshold
        # Weaviate uses cosine distance, which is 1 - cosine similarity
        distance_threshold = 1 - similarity_threshold if similarity_threshold > 0 else None

        near_vector = {
            "vector": query_vector,
        }
        if distance_threshold is not None:
            near_vector["distance"] = distance_threshold

        similar_documents = (
            self.storage_provider.client.query.get(
                self.index_name,
                ["embedding_pk", "metadata"],
            )
            .with_additional(["distance", "vector"])
            .with_near_vector(near_vector)
            .with_limit(limit)
            .do()
        )
        docs = similar_documents["data"]["Get"][self.index_name]
        for doc in docs:
            yield Document(
                embedding_pk=doc["embedding_pk"],
                metadata=json.loads(doc["metadata"]),
                vector=doc["_additional"]["vector"],
            )


class WeaviateVectorIndex(WeaviateIndexMixin, VectorIndex):
    pass


class WeaviateStorageProvider(StorageProvider[ProviderConfig, WeaviateIndexMixin]):
    config_class = ProviderConfig
    index_mixin = WeaviateIndexMixin

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        if self.config.API_KEY is None:
            raise ImproperlyConfigured("Weaviate API key is not set")
        auth_config = weaviate.auth.AuthApiKey(api_key=self.config.API_KEY)
        self.client = weaviate.Client(self.config.HOST, auth_client_secret=auth_config)
