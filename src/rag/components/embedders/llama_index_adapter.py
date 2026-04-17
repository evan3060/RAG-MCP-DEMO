"""
LlamaIndex 嵌入模型适配器

将我们的 BaseEmbedder 适配到 LlamaIndex 的 BaseEmbedding 接口
"""

from typing import List

from llama_index.core.embeddings import BaseEmbedding

from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder


class LlamaIndexEmbeddingAdapter(BaseEmbedding):
    """
    适配器：将 SiliconFlowEmbedder 包装为 LlamaIndex 兼容的嵌入模型

    【为什么需要适配器】
    LlamaIndex 的 Settings.embed_model 需要 BaseEmbedding 类型的对象
    我们的 SiliconFlowEmbedder 使用异步接口，需要同步包装
    """

    _embedder: SiliconFlowEmbedder = None

    def __init__(self, embedder: SiliconFlowEmbedder, **kwargs):
        # 先设置类变量，避免 pydantic 初始化问题
        self._embedder = embedder
        super().__init__(model_name=embedder.model_name or "siliconflow", **kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        import asyncio
        return asyncio.run(self._embedder.embed(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        import asyncio
        return asyncio.run(self._embedder.embed(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入向量"""
        import asyncio
        return asyncio.run(self._embedder.embed_batch(texts))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return await self._embedder.embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本的嵌入向量"""
        return await self._embedder.embed(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本的嵌入向量"""
        return await self._embedder.embed_batch(texts)
