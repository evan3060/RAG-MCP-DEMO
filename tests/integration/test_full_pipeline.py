"""集成测试 - 完整流程测试"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.rag.llamaindex.pipeline import RAGPipeline, SmartTextProcessor


class TestIntegrationEndToEnd:
    """端到端集成测试"""

    def test_text_processor_integration(self):
        """测试文本处理器集成"""
        processor = SmartTextProcessor()

        # 测试文本
        text = "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。" * 10

        # 处理文本
        nodes = processor.process(text, metadata={"source": "test"})

        # 验证
        assert len(nodes) > 0

    def test_chunking_quality(self):
        """测试分块质量"""
        processor = SmartTextProcessor(doc_type='general')

        # 长文本
        long_text = """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。
        它试图理解智能的本质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        
        机器学习是人工智能的一个子领域。
        机器学习专门研究计算机怎样模拟或实现人类的学习行为。
        机器学习是人工智能的核心。
        
        深度学习是机器学习的子领域。
        深度学习通过建立、模拟人脑进行分析学习的神经网络。
        """

        nodes = processor.process(long_text)

        # 验证分块质量
        assert len(nodes) >= 1

    def test_metadata_preservation(self):
        """测试元数据保留"""
        processor = SmartTextProcessor()

        text = "这是一段测试内容。"
        metadata = {
            "file_name": "test.txt",
            "file_type": "text",
            "source": "test"
        }

        nodes = processor.process(text, metadata=metadata)

        # 验证元数据被保留
        for node in nodes:
            assert node.metadata.get("file_name") == "test.txt"


class TestIntegrationHybridSearch:
    """混合检索集成测试"""

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_workflow(self):
        """测试混合检索工作流"""
        from llama_index.core.schema import TextNode, NodeWithScore
        from src.rag.llamaindex.hybrid_retriever import HybridRetriever

        # 创建模拟索引和检索器
        mock_index = MagicMock()
        nodes = [
            TextNode(text="关于机器学习的文档", metadata={"file_name": "ml.txt"}),
            TextNode(text="深度学习是机器学习的子领域", metadata={"file_name": "dl.txt"}),
        ]
        mock_index.docstore.docs = {node.node_id: node for node in nodes}

        # 创建向量检索器
        mock_vector_retriever = MagicMock()
        mock_vector_retriever.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[0], score=0.95),
            NodeWithScore(node=nodes[1], score=0.90),
        ])

        # 创建混合检索器
        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever,
            top_k=2
        )

        # 执行检索
        results = await retriever.aretrieve("机器学习")

        # 验证结果
        assert len(results) <= 2


class TestIntegrationErrorRecovery:
    """错误恢复集成测试"""

    def test_processor_with_invalid_input(self):
        """测试处理器处理无效输入"""
        processor = SmartTextProcessor()

        # 空字符串
        nodes = processor.process("")
        assert len(nodes) == 0

        # 仅空白字符
        nodes = processor.process("   \n\t   ")
        assert len(nodes) == 0

    def test_processor_with_special_characters(self):
        """测试处理器处理特殊字符"""
        processor = SmartTextProcessor()

        # 包含 emoji 和特殊符号
        text = "这是一个测试😊\n\n包含特殊字符：@#$%^&*()"
        nodes = processor.process(text)

        assert len(nodes) >= 1


class TestIntegrationPerformance:
    """性能集成测试"""

    def test_large_text_processing(self):
        """测试大文本处理性能"""
        processor = SmartTextProcessor()

        # 生成大文本
        base_text = "这是测试内容。" * 100
        large_text = base_text * 10  # 约 30KB 文本

        # 处理时间应该合理
        import time
        start = time.time()
        nodes = processor.process(large_text)
        elapsed = time.time() - start

        # 验证结果
        assert len(nodes) > 0
        assert elapsed < 5.0  # 5 秒超时


class TestIntegrationDataFlow:
    """数据流集成测试"""

    def test_document_to_nodes_flow(self):
        """测试文档到节点的数据流"""
        # 原始文档内容
        original_text = """
        人工智能是计算机科学的一个分支。

        机器学习是人工智能的核心。

        深度学习是机器学习的子领域，使用神经网络。
        """

        # 处理流程
        processor = SmartTextProcessor(doc_type='general')
        nodes = processor.process(original_text, metadata={"file_name": "ai.txt"})

        # 验证流程完整性
        assert len(nodes) > 0

        # 验证元数据保留
        for node in nodes:
            assert node.metadata.get("file_name") == "ai.txt"

    @pytest.mark.asyncio
    async def test_query_to_answer_flow(self):
        """测试查询到回答的数据流"""
        from llama_index.core.schema import TextNode, NodeWithScore
        from src.rag.llamaindex.hybrid_retriever import HybridRetriever

        # 模拟数据
        nodes = [
            TextNode(text="人工智能是AI的一个分支。", metadata={"file_name": "ai.txt"}),
            TextNode(text="机器学习是AI的核心技术。", metadata={"file_name": "ml.txt"}),
        ]

        mock_index = MagicMock()
        mock_index.docstore.docs = {node.node_id: node for node in nodes}

        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[0], score=0.95),
        ])

        # 检索
        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_retriever,
            top_k=2
        )

        results = await retriever.aretrieve("AI技术")

        # 验证结果
        assert len(results) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])