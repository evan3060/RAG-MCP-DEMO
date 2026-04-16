"""SiliconFlow 重排序实现"""

from typing import List, Dict, Any

import httpx

from src.rag.components.rerankers.base import BaseReranker, RerankResult


class SiliconFlowReranker(BaseReranker):
    """SiliconFlow 重排序实现"""

    API_BASE = "https://api.siliconflow.cn/v1"

    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "BAAI/bge-reranker-v2-m3")
        super().__init__(config)

    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("SiliconFlow Reranker 需要提供 api_key")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[RerankResult]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": self.top_n
                }
            )

            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                idx = item["index"]
                results.append(RerankResult(
                    id=doc_ids[idx],
                    text=documents[idx],
                    score=item["relevance_score"],
                    metadata=metadatas[idx]
                ))

            return results
