"""SiliconFlow 重排序实现"""

from typing import List, Dict, Any

import httpx
import json

from src.rag.components.rerankers.base import BaseReranker, RerankResult
from src.utils.registry import Registry


@Registry.register("reranker", "siliconflow")
@Registry.register("reranker", "openai")
@Registry.register("reranker", "myapi")
class SiliconFlowReranker(BaseReranker):
    """SiliconFlow 重排序实现"""

    API_BASE = "https://api.siliconflow.cn/v1"

    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "BAAI/bge-reranker-v2-m3")
        self.base_url = config.get("base_url", self.API_BASE)
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 确保 base_url 正确拼接
            base_url = self.base_url.rstrip('/')
            response = await client.post(
                f"{base_url}/v1/rerank",
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
            # 处理响应前面的空白字符
            text = response.text.strip()
            data = json.loads(text)

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
