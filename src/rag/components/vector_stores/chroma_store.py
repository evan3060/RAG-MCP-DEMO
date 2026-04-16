"""Chroma 向量数据库实现 - 本地嵌入式存储"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.rag.components.vector_stores.base import BaseVectorStore, VectorSearchResult


class ChromaVectorStore(BaseVectorStore):
    """Chroma 本地向量数据库实现"""

    def __init__(self, config: dict):
        self.persist_dir = config.get("persist_directory", "./data/chroma_db")
        self.distance_fn = config.get("distance_fn", "cosine")
        super().__init__(config)

    def _validate_config(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        if not hasattr(self, '_client'):
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn}
            )
        return self._collection

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        collection = self._get_client()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        collection = self._get_client()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            search_results.append(VectorSearchResult(
                id=doc_id,
                score=results["distances"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}
            ))
        return search_results
