"""
组件自动注册

导入所有组件以触发注册装饰器
"""

from src.rag.components.llms.qianfan_llm import QianfanLLM
from src.rag.components.llms.openai_compatible_llm import OpenAICompatibleLLM
from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder
from src.rag.components.rerankers.siliconflow_reranker import SiliconFlowReranker


__all__ = [
    "QianfanLLM",
    "OpenAICompatibleLLM",
    "SiliconFlowEmbedder",
    "SiliconFlowReranker",
]
