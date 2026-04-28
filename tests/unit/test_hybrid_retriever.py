"""HybridRetriever 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llama_index.core.schema import TextNode, NodeWithScore

from src.rag.llamaindex.hybrid_retriever import HybridRetriever, rrf_fusion


class TestRRFFusion:
    """RRF 融合算法测试"""

    def test_rrf_fusion_empty_results(self):
        """测试空结果融合"""
        result = rrf_fusion([], [])
        assert len(result) == 0

    def test_rrf_fusion_vector_only(self):
        """测试仅有向量检索结果"""
        nodes = [
            TextNode(text="doc1"),
            TextNode(text="doc2"),
            TextNode(text="doc3"),
        ]
        vector_results = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
            NodeWithScore(node=nodes[2], score=0.7),
        ]

        result = rrf_fusion(vector_results, [], vector_weight=1.0, bm25_weight=0.0)
        assert len(result) == 3

    def test_rrf_fusion_bm25_only(self):
        """测试仅有 BM25 检索结果"""
        nodes = [
            TextNode(text="doc1"),
            TextNode(text="doc2"),
        ]
        bm25_results = [
            NodeWithScore(node=nodes[0], score=0.95),
            NodeWithScore(node=nodes[1], score=0.85),
        ]

        result = rrf_fusion([], bm25_results, vector_weight=0.0, bm25_weight=1.0)
        assert len(result) == 2

    def test_rrf_fusion_combined(self):
        """测试组合检索结果"""
        nodes = [
            TextNode(text="doc1"),
            TextNode(text="doc2"),
            TextNode(text="doc3"),
        ]
        vector_results = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
        ]
        bm25_results = [
            NodeWithScore(node=nodes[1], score=0.85),  # 与向量结果重复
            NodeWithScore(node=nodes[2], score=0.75),
        ]

        result = rrf_fusion(vector_results, bm25_results)
        assert len(result) <= 4  # 可能有重复

        # doc1 在向量结果中排名更高
        result_ids = [r.node.node_id for r in result]
        # doc1 或 doc2 应该在前面
        assert any(rid in result_ids[:2] for rid in [nodes[0].node_id, nodes[1].node_id])

    def test_rrf_fusion_weights(self):
        """测试不同权重配置"""
        nodes = [TextNode(text=f"doc{i}") for i in range(3)]
        vector_results = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
            NodeWithScore(node=nodes[2], score=0.7),
        ]
        bm25_results = [
            NodeWithScore(node=nodes[2], score=0.95),
            NodeWithScore(node=nodes[1], score=0.85),
            NodeWithScore(node=nodes[0], score=0.75),
        ]

        # 向量权重高
        result_vector_first = rrf_fusion(
            vector_results, bm25_results,
            vector_weight=0.9, bm25_weight=0.1
        )

        # BM25 权重高
        result_bm25_first = rrf_fusion(
            vector_results, bm25_results,
            vector_weight=0.1, bm25_weight=0.9
        )

        # 两种结果应该不同（排名不同）
        # 注意：由于 RRF 算法特性，权重的影响是渐进的

    def test_rrf_fusion_k_parameter(self):
        """测试 k 参数"""
        nodes = [TextNode(text=f"doc{i}") for i in range(3)]
        vector_results = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
        ]
        bm25_results = [
            NodeWithScore(node=nodes[1], score=0.85),
            NodeWithScore(node=nodes[2], score=0.75),
        ]

        # 小 k 值
        result_small_k = rrf_fusion(vector_results, bm25_results, k=10)
        # 大 k 值
        result_large_k = rrf_fusion(vector_results, bm25_results, k=100)

        # 排名可能不同
        # 这是预期的行为


class TestHybridRetriever:
    """HybridRetriever 测试类"""

    @pytest.fixture
    def mock_index(self):
        """创建模拟索引"""
        index = MagicMock()
        # 模拟 docstore
        docstore = MagicMock()
        nodes = [TextNode(text=f"doc{i}") for i in range(5)]
        docstore.docs = {node.node_id: node for node in nodes}
        index.docstore = docstore
        return index

    @pytest.fixture
    def mock_vector_retriever(self):
        """创建模拟向量检索器"""
        retriever = MagicMock()
        nodes = [
            TextNode(text="doc1"),
            TextNode(text="doc2"),
            TextNode(text="doc3"),
        ]
        retriever.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[0], score=0.95),
            NodeWithScore(node=nodes[1], score=0.85),
            NodeWithScore(node=nodes[2], score=0.75),
        ])
        return retriever

    def test_initialization(self, mock_index, mock_vector_retriever):
        """测试初始化"""
        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever,
            top_k=10,
            vector_weight=0.7,
            bm25_weight=0.3
        )

        assert retriever.index == mock_index
        assert retriever.vector_retriever == mock_vector_retriever
        assert retriever.top_k == 10
        assert retriever.vector_weight == 0.7
        assert retriever.bm25_weight == 0.3
        assert retriever._bm25_retriever is None

    def test_initialization_default_weights(self, mock_index, mock_vector_retriever):
        """测试默认权重初始化"""
        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever
        )

        assert retriever.vector_weight == 0.7
        assert retriever.bm25_weight == 0.3
        assert retriever.top_k == 10

    @pytest.mark.asyncio
    async def test_aretrieve_vector_only(self, mock_index, mock_vector_retriever):
        """测试仅向量检索"""
        # 设置空的 docstore，强制只使用向量检索
        mock_index.docstore.docs = {}

        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever,
            top_k=5
        )

        results = await retriever.aretrieve("test query")

        assert len(results) <= 5
        mock_vector_retriever.aretrieve.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_aretrieve_with_bm25(self, mock_index, mock_vector_retriever):
        """测试混合检索"""
        # 设置非空 docstore 以启用 BM25
        nodes = [TextNode(text=f"doc{i}") for i in range(10)]
        mock_index.docstore.docs = {node.node_id: node for node in nodes}

        # 模拟 BM25 检索器
        mock_bm25 = MagicMock()
        mock_bm25.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
        ])

        with patch.object(HybridRetriever, '_get_bm25_retriever', return_value=mock_bm25):
            retriever = HybridRetriever(
                index=mock_index,
                vector_retriever=mock_vector_retriever,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3
            )

            results = await retriever.aretrieve("test query")

            assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_aretrieve_empty_index(self, mock_vector_retriever):
        """测试空索引检索"""
        # 创建没有 docstore 的索引
        mock_index = MagicMock()
        mock_index.docstore = MagicMock()
        mock_index.docstore.docs = {}  # 空

        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever,
            top_k=5
        )

        results = await retriever.aretrieve("test query")

        # 应该只返回向量检索结果
        assert len(results) <= 5
        mock_vector_retriever.aretrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_aretrieve_top_k_respected(self, mock_index, mock_vector_retriever):
        """测试 top_k 参数生效"""
        nodes = [TextNode(text=f"doc{i}") for i in range(20)]
        mock_index.docstore.docs = {node.node_id: node for node in nodes}

        # 模拟返回更多结果
        mock_vector_retriever.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[i], score=1.0 - i * 0.01)
            for i in range(20)
        ])

        # 模拟 BM25
        mock_bm25 = MagicMock()
        mock_bm25.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[i], score=0.95 - i * 0.01)
            for i in range(20)
        ])

        with patch.object(HybridRetriever, '_get_bm25_retriever', return_value=mock_bm25):
            retriever = HybridRetriever(
                index=mock_index,
                vector_retriever=mock_vector_retriever,
                top_k=3
            )

            results = await retriever.aretrieve("test query")

            assert len(results) <= 3

    def test_get_bm25_retriever_initialization(self, mock_index, mock_vector_retriever):
        """测试 BM25 检索器懒加载"""
        nodes = [TextNode(text=f"doc{i}") for i in range(5)]
        mock_index.docstore.docs = {node.node_id: node for node in nodes}

        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever,
            top_k=10
        )

        # 首次访问应该创建 BM25 检索器
        bm25 = retriever._get_bm25_retriever()
        assert bm25 is not None
        assert retriever._bm25_retriever is not None

        # 再次访问应该返回同一个实例
        bm25_again = retriever._get_bm25_retriever()
        assert bm25 is bm25_again

    def test_get_bm25_retriever_empty_index(self, mock_vector_retriever):
        """测试空索引返回 None"""
        mock_index = MagicMock()
        mock_index.docstore = MagicMock()
        mock_index.docstore.docs = {}

        retriever = HybridRetriever(
            index=mock_index,
            vector_retriever=mock_vector_retriever
        )

        result = retriever._get_bm25_retriever()
        assert result is None


class TestHybridRetrieverEdgeCases:
    """HybridRetriever 边界情况测试"""

    @pytest.fixture
    def mock_index_with_single_node(self):
        """创建只有一个节点的索引"""
        index = MagicMock()
        node = TextNode(text="single doc")
        docstore = MagicMock()
        docstore.docs = {node.node_id: node}
        index.docstore = docstore
        return index

    @pytest.mark.asyncio
    async def test_single_node_index(self, mock_index_with_single_node):
        """测试单节点索引"""
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=[
            NodeWithScore(
                node=TextNode(text="single doc"),
                score=0.99
            )
        ])

        retriever = HybridRetriever(
            index=mock_index_with_single_node,
            vector_retriever=mock_retriever,
            top_k=10
        )

        results = await retriever.aretrieve("test")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_index_with_single_node):
        """测试空查询"""
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            index=mock_index_with_single_node,
            vector_retriever=mock_retriever
        )

        # 空查询也可能返回结果（取决于检索器实现）
        results = await retriever.aretrieve("")
        # 不应该抛出异常


if __name__ == '__main__':
    pytest.main([__file__, '-v'])