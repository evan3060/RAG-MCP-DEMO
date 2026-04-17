"""LlamaIndex RAG Pipeline - 完整的检索增强生成流程"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder
from src.rag.components.vector_stores.chroma_store import ChromaVectorStore
from src.rag.components.rerankers.siliconflow_reranker import SiliconFlowReranker
from src.rag.components.llms.qianfan_llm import QianfanLLM
from src.rag.components.llms.siliconflow_llm import SiliconFlowLLM
from src.rag.llamaindex.hybrid_retriever import HybridRetriever


class RAGPipeline:
    """RAG Pipeline 主类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index: Optional[VectorStoreIndex] = None
        self._configure_settings()

    def _configure_settings(self):
        """配置 LlamaIndex 全局设置"""
        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "qianfan")

        embedding_config = self.config.get("embedding", {})
        embed_provider = embedding_config.get("provider", "siliconflow")

        # 配置嵌入模型
        if embed_provider == "siliconflow":
            sf_config = embedding_config.get("siliconflow", {})
            Settings.embed_model = SiliconFlowEmbedder({
                "api_key": sf_config.get("api_key"),
                "model": sf_config.get("model", "BAAI/bge-large-zh-v1.5"),
                "base_url": sf_config.get("base_url")
            })

        # 配置 LLM
        if llm_provider == "qianfan":
            qf_config = llm_config.get("qianfan", {})
            Settings.llm = QianfanLLM({
                "api_key": qf_config.get("api_key"),
                "secret_key": qf_config.get("secret_key"),
                "model": qf_config.get("model", "ERNIE-Bot-4"),
                "base_url": qf_config.get("base_url")
            })
        elif llm_provider == "siliconflow":
            sf_config = llm_config.get("siliconflow", {})
            Settings.llm = SiliconFlowLLM({
                "api_key": sf_config.get("api_key"),
                "model": sf_config.get("model", "deepseek-ai/DeepSeek-V3"),
                "base_url": sf_config.get("base_url")
            })

    async def build_index(self, documents_path: str) -> VectorStoreIndex:
        """构建知识库索引"""
        documents = SimpleDirectoryReader(documents_path).load_data()

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )
        nodes = splitter.get_nodes_from_documents(documents)

        vector_store = ChromaVectorStore({
            "persist_directory": "./data/chroma_db",
            "collection_name": "knowledge_base"
        })
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context
        )

        return self.index

    async def ask(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """智能问答"""
        if not self.index:
            raise ValueError("索引未构建，请先调用 build_index()")

        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20
        )
        hybrid_retriever = HybridRetriever(
            index=self.index,
            vector_retriever=vector_retriever,
            top_k=10
        )

        chat_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000)
        )

        response = await chat_engine.achat(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata
            })

        return {
            "answer": response.response,
            "sources": sources,
            "session_id": session_id or "new_session"
        }

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """纯检索（不调用 LLM）"""
        if not self.index:
            raise ValueError("索引未构建")

        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        nodes = await vector_retriever.aretrieve(query)

        return [
            {
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata
            }
            for node in nodes
        ]
