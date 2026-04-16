"""重排序基类 - 对初步检索结果进行精排"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RerankResult:
    """重排序结果"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class BaseReranker(ABC):
    """重排序器抽象基类"""

    def __init__(self, config: dict):
        self.config = config
        self.top_n = config.get("top_n", 5)
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[RerankResult]:
        pass
