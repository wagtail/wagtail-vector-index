import json
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import weaviate
from django.core.exceptions import ImproperlyConfigured

from wagtail_vector_index.backends.base import Backend, Index
from wagtail_vector_index.index.base import Document


@dataclass
class BackendConfig:
    HOST: str
    API_KEY: str | None = None


class WeaviateIndex(Index):
    def __init__(self, index_name: str, api_client: weaviate.Client, **kwargs: Any):
        self.index_name = index_name
        self.client = api_client

    def upsert(self, *, documents: Iterable[Document]) -> None:
        with self.client.batch as batch:
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

    def similarity_search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> Generator[Document, None, None]:
        near_vector = {
            "vector": query_vector,
        }
        similar_documents = (
            self.client.query.get(
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


class WeaviateBackend(Backend[BackendConfig, WeaviateIndex]):
    config_class = BackendConfig

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        if self.config.API_KEY is None:
            raise ImproperlyConfigured("Weaviate API key is not set")
        auth_config = weaviate.auth.AuthApiKey(api_key=self.config.API_KEY)
        self.client = weaviate.Client(self.config.HOST, auth_client_secret=auth_config)

    def get_index(self, index_name: str) -> WeaviateIndex:
        return WeaviateIndex(index_name, api_client=self.client)

    def create_index(self, index_name: str, **kwargs: Any) -> WeaviateIndex:
        self.client.schema.create_class(
            {
                "class": index_name,
            }
        )
        return self.get_index(index_name)

    def delete_index(self, index_name):
        self.client.schema.delete_class(index_name)
