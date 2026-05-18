"""Vector Database Bridge for SMGP.

Provides VectorDBSyncer, a bridge that synchronizes graph nodes with external
vector databases (Qdrant, Pinecone, Weaviate) for scalable similarity search.

References:
  - Qdrant: https://qdrant.tech/
  - Pinecone: https://www.pinecone.io/
  - Weaviate: https://weaviate.io/
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("qdrant", "pinecone", "weaviate")


class VectorDBSyncer:
    """Synchronizes SMGP graph nodes with an external vector database.

    Supports three backends: Qdrant, Pinecone, and Weaviate. The syncer lazily
    imports the client library only when needed, so dependencies are optional.

    Attributes:
        graph: The SpectralMemoryGraph to sync.
        backend: Name of the vector DB backend.
        client: The initialized vector DB client (lazy).
    """

    def __init__(
        self,
        graph: Any,
        backend: str = "qdrant",
        **kwargs: Any,
    ) -> None:
        """Initialize the vector DB syncer.

        Args:
            graph: A SpectralMemoryGraph instance.
            backend: Vector DB backend name: "qdrant", "pinecone", or "weaviate".
            **kwargs: Backend-specific configuration options.
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported: {SUPPORTED_BACKENDS}"
            )
        self.graph = graph
        self.backend = backend
        self._client: Any = None
        self._kwargs = kwargs

    def _get_client(self) -> Any:
        """Lazy-initialize and return the vector DB client.

        Returns:
            The initialized client for the configured backend.

        Raises:
            ImportError: If the backend's client library is not installed.
        """
        if self._client is not None:
            return self._client

        if self.backend == "qdrant":
            self._client = self._init_qdrant(**self._kwargs)
        elif self.backend == "pinecone":
            self._client = self._init_pinecone(**self._kwargs)
        elif self.backend == "weaviate":
            self._client = self._init_weaviate(**self._kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._client

    def _init_qdrant(self, **kwargs: Any) -> Any:
        """Initialize Qdrant client.

        For testing, uses :memory: mode by default.

        Args:
            **kwargs: Options passed to QdrantClient. If no 'location' is
                specified, defaults to ':memory:' for in-memory testing.

        Returns:
            QdrantClient instance.

        Raises:
            ImportError: If qdrant-client is not installed.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for Qdrant backend. "
                "Install with: pip install qdrant-client"
            ) from e

        location = kwargs.pop("location", ":memory:")
        client = QdrantClient(location=location, **kwargs)
        logger.info("Initialized Qdrant client (location=%s)", location)
        return client

    def _init_pinecone(self, **kwargs: Any) -> Any:
        """Initialize Pinecone client.

        Args:
            **kwargs: Options including 'api_key' and 'environment'.

        Returns:
            Pinecone index instance.

        Raises:
            ImportError: If pinecone-client is not installed.
        """
        try:
            import pinecone
        except ImportError as e:
            raise ImportError(
                "pinecone-client is required for Pinecone backend. "
                "Install with: pip install pinecone-client"
            ) from e

        api_key = kwargs.get("api_key", "")
        environment = kwargs.get("environment", "")
        if api_key and environment:
            pinecone.init(api_key=api_key, environment=environment)
        logger.info("Initialized Pinecone client")
        return pinecone

    def _init_weaviate(self, **kwargs: Any) -> Any:
        """Initialize Weaviate client.

        Args:
            **kwargs: Options passed to weaviate.Client.

        Returns:
            Weaviate client instance.

        Raises:
            ImportError: If weaviate-client is not installed.
        """
        try:
            import weaviate
        except ImportError as e:
            raise ImportError(
                "weaviate-client is required for Weaviate backend. "
                "Install with: pip install weaviate-client"
            ) from e

        url = kwargs.pop("url", "http://localhost:8080")
        client = weaviate.Client(url=url, **kwargs)
        logger.info("Initialized Weaviate client (url=%s)", url)
        return client

    def sync_nodes(self, collection_name: str = "smgp_nodes") -> dict[str, Any]:
        """Upload all graph nodes with their HD vectors to the vector DB.

        Creates the collection if it doesn't exist, then upserts all nodes.

        Args:
            collection_name: Name of the collection/index to sync into.

        Returns:
            Dict with sync statistics: {num_synced, collection_name}.
        """
        if self.backend == "qdrant":
            return self._sync_qdrant(collection_name)
        elif self.backend == "pinecone":
            return self._sync_pinecone(collection_name)
        elif self.backend == "weaviate":
            return self._sync_weaviate(collection_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _sync_qdrant(self, collection_name: str) -> dict[str, Any]:
        """Sync nodes to Qdrant.

        Args:
            collection_name: Qdrant collection name.

        Returns:
            Dict with sync statistics.
        """
        from qdrant_client.models import (
            Distance,
            PointStruct,
            VectorParams,
        )

        client = self._get_client()

        # Create collection if not exists
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.graph.hd.dim,
                    distance=Distance.COSINE,
                ),
            )

        # Upsert all nodes
        points = []
        node_list = self.graph.nodes()
        for idx, node_id in enumerate(node_list):
            node_data = self.graph.get_node(node_id)
            if node_data is None or node_data.get("vector") is None:
                continue
            vector = node_data["vector"].astype(np.float32).tolist()
            payload = {
                "node_id": node_id,
                "label": node_data.get("label", ""),
                "properties": node_data.get("properties", {}),
            }
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )

        logger.info(
            "Synced %d nodes to Qdrant collection '%s'",
            len(points),
            collection_name,
        )
        return {
            "num_synced": len(points),
            "collection_name": collection_name,
        }

    def _sync_pinecone(self, collection_name: str) -> dict[str, Any]:
        """Sync nodes to Pinecone.

        Args:
            collection_name: Pinecone index name.

        Returns:
            Dict with sync statistics.
        """
        client = self._get_client()
        # Pinecone index management
        if collection_name not in client.list_indexes():
            client.create_index(
                name=collection_name,
                dimension=self.graph.hd.dim,
                metric="cosine",
            )
        index = client.Index(collection_name)

        vectors = []
        node_list = self.graph.nodes()
        for idx, node_id in enumerate(node_list):
            node_data = self.graph.get_node(node_id)
            if node_data is None or node_data.get("vector") is None:
                continue
            vector = node_data["vector"].astype(np.float32).tolist()
            vectors.append((str(idx), vector, {"node_id": node_id}))

        if vectors:
            index.upsert(vectors=vectors)

        logger.info(
            "Synced %d nodes to Pinecone index '%s'",
            len(vectors),
            collection_name,
        )
        return {
            "num_synced": len(vectors),
            "collection_name": collection_name,
        }

    def _sync_weaviate(self, collection_name: str) -> dict[str, Any]:
        """Sync nodes to Weaviate.

        Args:
            collection_name: Weaviate class name.

        Returns:
            Dict with sync statistics.
        """
        client = self._get_client()

        class_obj = {
            "class": collection_name,
            "vectorizer": "none",
            "properties": [
                {"name": "node_id", "dataType": ["string"]},
                {"name": "label", "dataType": ["string"]},
            ],
        }
        if not client.schema.exists(collection_name):
            client.schema.create_class(class_obj)

        client.batch.configure(batch_size=100)
        count = 0
        with client.batch as _batch:
            node_list = self.graph.nodes()
            for node_id in node_list:
                node_data = self.graph.get_node(node_id)
                if node_data is None or node_data.get("vector") is None:
                    continue
                vector = node_data["vector"].astype(np.float32).tolist()
                client.batch.add_data_object(
                    data_object={
                        "node_id": node_id,
                        "label": node_data.get("label", ""),
                    },
                    class_name=collection_name,
                    vector=vector,
                )
                count += 1

        logger.info(
            "Synced %d nodes to Weaviate class '%s'",
            count,
            collection_name,
        )
        return {
            "num_synced": count,
            "collection_name": collection_name,
        }

    def query_external(
        self,
        query_vector: np.ndarray,
        collection_name: str = "smgp_nodes",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the external vector DB for similar vectors.

        Args:
            query_vector: Query HD vector.
            collection_name: Collection/index to search.
            k: Number of results.

        Returns:
            List of result dicts with node_id, score, and payload.
        """
        if self.backend == "qdrant":
            return self._query_qdrant(query_vector, collection_name, k)
        elif self.backend == "pinecone":
            return self._query_pinecone(query_vector, collection_name, k)
        elif self.backend == "weaviate":
            return self._query_weaviate(query_vector, collection_name, k)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _query_qdrant(
        self,
        query_vector: np.ndarray,
        collection_name: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Query Qdrant for similar vectors.

        Args:
            query_vector: Query vector.
            collection_name: Collection name.
            k: Number of results.

        Returns:
            List of result dicts.
        """
        client = self._get_client()
        query_vec = query_vector.astype(np.float32).tolist()

        results = client.query_points(
            collection_name=collection_name,
            query=query_vec,
            limit=k,
        ).points

        return [
            {
                "node_id": hit.payload.get("node_id", ""),
                "score": hit.score,
                "label": hit.payload.get("label", ""),
                "properties": hit.payload.get("properties", {}),
            }
            for hit in results
        ]

    def _query_pinecone(
        self,
        query_vector: np.ndarray,
        collection_name: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Query Pinecone for similar vectors.

        Args:
            query_vector: Query vector.
            collection_name: Index name.
            k: Number of results.

        Returns:
            List of result dicts.
        """
        client = self._get_client()
        index = client.Index(collection_name)
        query_vec = query_vector.astype(np.float32).tolist()

        results = index.query(vector=query_vec, top_k=k, include_metadata=True)

        return [
            {
                "node_id": match.metadata.get("node_id", ""),
                "score": match.score,
            }
            for match in results.matches
        ]

    def _query_weaviate(
        self,
        query_vector: np.ndarray,
        collection_name: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Query Weaviate for similar vectors.

        Args:
            query_vector: Query vector.
            collection_name: Class name.
            k: Number of results.

        Returns:
            List of result dicts.
        """
        client = self._get_client()
        query_vec = query_vector.astype(np.float32).tolist()

        results = (
            client.query
            .get(collection_name, ["node_id", "label"])
            .with_near_vector({"vector": query_vec})
            .with_limit(k)
            .do()
        )

        output = []
        for item in results.get("data", {}).get("Get", {}).get(collection_name, []):
            output.append({
                "node_id": item.get("node_id", ""),
                "score": item.get("_additional", {}).get("distance", 0.0),
                "label": item.get("label", ""),
            })
        return output

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        text_filter: str,
        collection_name: str = "smgp_nodes",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Combine vector search with metadata filtering.

        Args:
            query_vector: Query HD vector.
            text_filter: Text to filter by (matches against node label or properties).
            collection_name: Collection/index to search.
            k: Number of results.

        Returns:
            List of result dicts matching both vector similarity and filter.
        """
        # Get all results from vector search, then filter
        all_results = self.query_external(
            query_vector, collection_name, k=k * 10  # Oversample for filtering
        )

        text_lower = text_filter.lower()
        filtered = [
            r for r in all_results
            if text_lower in r.get("label", "").lower()
            or text_lower in str(r.get("properties", {})).lower()
            or text_lower in r.get("node_id", "").lower()
        ]

        return filtered[:k]
