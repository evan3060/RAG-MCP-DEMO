"""向量数据库基类 - 存储和检索文本向量"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    text: str
    metadata: Dict


class BaseVectorStore(ABC):
    """向量数据库抽象基类"""

    def __init__(self, config: dict):
        self.config = config
        self.collection_name = config.get("collection_name", "default")
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        pass
