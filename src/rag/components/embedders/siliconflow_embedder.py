"""
SiliconFlow 嵌入模型实现

【推荐模型】
- BAAI/bge-large-zh-v1.5: 中文最佳（1024维）
- BAAI/bge-m3: 多语言支持
- BAAI/bge-small-zh-v1.5: 轻量快速

【API 文档】
https://docs.siliconflow.cn/api-reference/embeddings/create-embeddings
"""

from typing import List

import httpx

from src.rag.components.embedders.base import BaseEmbedder


class SiliconFlowEmbedder(BaseEmbedder):
    """SiliconFlow 嵌入模型实现"""

    API_BASE = "https://api.siliconflow.cn/v1"

    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        super().__init__(config)

    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("SiliconFlow Embedder 需要提供 api_key")

    async def embed(self, text: str) -> List[float]:
        """嵌入单条文本"""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name or "BAAI/bge-large-zh-v1.5",
                    "input": texts,
                    "encoding_format": "float"
                }
            )

            response.raise_for_status()
            data = response.json()

            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
