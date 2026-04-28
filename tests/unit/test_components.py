"""组件基类测试"""

import pytest
from abc import ABC
from typing import List
from llama_index.core.schema import TextNode, Document


class TestComponentContracts:
    """组件接口契约测试"""

    def test_component_modules_exist(self):
        """测试组件模块存在"""
        from src.rag.components.llms import base as llm_base
        from src.rag.components.embedders import base as embedder_base
        from src.rag.components.vector_stores import base as vs_base
        from src.rag.components.loaders import base as loader_base
        
        assert llm_base is not None
        assert embedder_base is not None
        assert vs_base is not None
        assert loader_base is not None

    def test_llm_base_import(self):
        """测试 LLM 基类可导入"""
        from src.rag.components.llms.base import BaseLLM
        assert BaseLLM is not None

    def test_embedder_base_import(self):
        """测试 Embedder 基类可导入"""
        from src.rag.components.embedders.base import BaseEmbedder
        assert BaseEmbedder is not None

    def test_vector_store_base_import(self):
        """测试 VectorStore 基类可导入"""
        from src.rag.components.vector_stores.base import BaseVectorStore
        assert BaseVectorStore is not None

    def test_loader_base_import(self):
        """测试 Loader 基类可导入"""
        from src.rag.components.loaders.base import BaseLoader
        assert BaseLoader is not None

    def test_llm_implementations_exist(self):
        """测试 LLM 实现类存在"""
        from src.rag.components.llms.qianfan_llm import QianfanLLM
        from src.rag.components.llms.siliconflow_llm import SiliconFlowLLM
        from src.rag.components.llms.openai_compatible_llm import OpenAICompatibleLLM
        
        assert QianfanLLM is not None
        assert SiliconFlowLLM is not None
        assert OpenAICompatibleLLM is not None

    def test_embedder_implementations_exist(self):
        """测试 Embedder 实现类存在"""
        from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder
        
        assert SiliconFlowEmbedder is not None

    def test_reranker_implementations_exist(self):
        """测试 Reranker 实现类存在"""
        from src.rag.components.rerankers.siliconflow_reranker import SiliconFlowReranker
        
        assert SiliconFlowReranker is not None

    def test_vector_store_implementations_exist(self):
        """测试 VectorStore 实现类存在"""
        from src.rag.components.vector_stores.chroma_store import ChromaVectorStore
        
        assert ChromaVectorStore is not None

    def test_loader_implementations_exist(self):
        """测试 Loader 实现类存在"""
        from src.rag.components.loaders.pdf_loader import PDFLoader
        from src.rag.components.loaders.office_loader import DocxLoader, ExcelLoader
        
        assert PDFLoader is not None
        assert DocxLoader is not None
        assert ExcelLoader is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])