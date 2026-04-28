"""
SiliconFlow 嵌入模型实现 - LlamaIndex BaseEmbedding 子类

【推荐模型】
- BAAI/bge-large-zh-v1.5: 中文最佳（1024维）
- BAAI/bge-m3: 多语言支持
- BAAI/bge-small-zh-v1.5: 轻量快速

【API 文档】
https://docs.siliconflow.cn/api-reference/embeddings/create-embeddings
"""

from typing import List

import httpx
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class SiliconFlowEmbedder(BaseEmbedding):
    """
    SiliconFlow 嵌入模型实现

    继承 LlamaIndex 的 BaseEmbedding，可直接用于 Settings.embed_model
    """

    # 使用 PrivateAttr 存储配置，避免 Pydantic 处理
    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr(default="https://api.siliconflow.cn/v1")
    _model: str = PrivateAttr(default="BAAI/bge-large-zh-v1.5")

    def __init__(self, config: dict, **kwargs):
        """初始化嵌入模型"""
        # 先提取配置
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "https://api.siliconflow.cn/v1")
        model = config.get("model", "BAAI/bge-large-zh-v1.5")

        if not api_key:
            raise ValueError("SiliconFlow Embedder 需要提供 api_key")

        # 调用父类初始化
        super().__init__(model_name=model, **kwargs)

        # 使用 PrivateAttr 存储（在 super().__init__ 之后）
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        return self._embed_sync(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        return self._embed_sync(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入向量"""
        return self._embed_batch_sync(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return await self._embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本的嵌入向量"""
        return await self._embed(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本的嵌入向量"""
        return await self._embed_batch(texts)

    def _embed_sync(self, text: str) -> List[float]:
        """同步嵌入单条文本"""
        results = self._embed_batch_sync([text])
        return results[0]

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """同步批量嵌入文本"""
        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self._model,
                    "input": texts,
                    "encoding_format": "float"
                },
                timeout=120.0
            )

            response.raise_for_status()
            data = response.json()

            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

    async def _embed(self, text: str) -> List[float]:
        """嵌入单条文本"""
        results = await self._embed_batch([text])
        return results[0]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self._model,
                    "input": texts,
                    "encoding_format": "float"
                },
                timeout=120.0
            )

            response.raise_for_status()
            data = response.json()

            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
