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
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaIndexChromaStore

from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder
from src.rag.components.rerankers.siliconflow_reranker import SiliconFlowReranker
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

        # 配置嵌入模型 (直接继承 BaseEmbedding)
        if embed_provider == "siliconflow":
            sf_config = embedding_config.get("siliconflow", {})
            Settings.embed_model = SiliconFlowEmbedder({
                "api_key": sf_config.get("api_key"),
                "model": sf_config.get("model", "BAAI/bge-large-zh-v1.5"),
                "base_url": sf_config.get("base_url")
            })

        # 配置 LLM (SiliconFlow 使用自定义实现，千帆待实现)
        if llm_provider == "siliconflow":
            sf_config = llm_config.get("siliconflow", {})
            # 暂时使用 SiliconFlow 的直接实现
            # 实际使用时需要创建 LlamaIndex 兼容的包装器
            self._llm_config = sf_config

    async def build_index(self, documents_path: str) -> VectorStoreIndex:
        """构建知识库索引"""
        documents = SimpleDirectoryReader(documents_path).load_data()

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)

        # 使用 Chroma PersistentClient
        import chromadb
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        vector_store = LlamaIndexChromaStore(
            chroma_client=chroma_client,
            collection_name="knowledge_base"
        )
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
