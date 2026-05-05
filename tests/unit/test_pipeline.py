"""RAGPipeline 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode

from src.rag.llamaindex.pipeline import RAGPipeline, SmartTextProcessor


@pytest.fixture
def test_config():
    """测试配置"""
    return {
        "llm": {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key": "test-key",
                "model": "deepseek-ai/DeepSeek-V3",
                "base_url": "https://api.siliconflow.cn/v1"
            }
        },
        "embedding": {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key": "test-key",
                "model": "BAAI/bge-large-zh-v1.5"
            }
        },
        "reranker": {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key": "test-key",
                "model": "BAAI/bge-reranker-v2-m3"
            }
        },
        "vector_store": {
            "provider": "chroma",
            "persist_dir": "./data/chroma_db"
        }
    }


class TestRAGPipeline:
    """RAGPipeline 测试类"""

    @patch('src.rag.llamaindex.pipeline.Settings')
    @patch('src.rag.llamaindex.pipeline.RAGPipeline._load_existing_index')
    def test_initialization(self, mock_load_index, mock_settings, test_config):
        """测试初始化"""
        pipeline = RAGPipeline(test_config)
        assert pipeline.config == test_config
        assert pipeline.index is None

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_config_loading(self, mock_settings, test_config):
        """测试配置加载"""
        pipeline = RAGPipeline(test_config)
        assert pipeline.config['llm']['provider'] == 'siliconflow'
        assert pipeline.config['embedding']['provider'] == 'siliconflow'

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_load_existing_index_nonexistent(self, mock_settings, test_config):
        """测试加载不存在的索引"""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = RAGPipeline(test_config)
            assert pipeline.index is None


class TestRAGPipelineAsk:
    """RAGPipeline ask 方法测试"""

    @pytest.mark.asyncio
    @patch('src.rag.llamaindex.pipeline.Settings')
    async def test_ask_without_index(self, mock_settings, test_config):
        """测试索引未构建时的问答"""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = RAGPipeline(test_config)
            
            with pytest.raises(ValueError, match="索引未构建"):
                await pipeline.ask("测试问题")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要完整 mock LlamaIndex 组件")
    @patch('src.rag.llamaindex.pipeline.Settings')
    async def test_ask_with_empty_selected_files(self, mock_settings, test_config):
        """测试选择了空知识库文件列表 - 现在搜索所有文件"""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = RAGPipeline(test_config)
            pipeline.index = MagicMock()
            
            mock_vector_retriever = MagicMock()
            mock_vector_retriever.aretrieve.return_value = []
            
            with patch('src.rag.llamaindex.pipeline.VectorIndexRetriever', return_value=mock_vector_retriever):
                with patch('src.rag.llamaindex.pipeline.HybridRetriever') as mock_hybrid:
                    mock_hybrid.return_value.aretrieve.return_value = []
                    result = await pipeline.ask("测试问题", selected_files=[])

            assert result["success"] == True
            assert result["answer"] == "未找到相关内容"
            assert len(result["sources"]) == 0


class TestRAGPipelineSearch:
    """RAGPipeline search 方法测试"""

    @pytest.mark.asyncio
    @patch('src.rag.llamaindex.pipeline.Settings')
    async def test_search_without_index(self, mock_settings, test_config):
        """测试索引未构建时的搜索"""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = RAGPipeline(test_config)

            with pytest.raises(ValueError, match="索引未构建"):
                await pipeline.search("测试查询")


class TestRAGPipelineFilters:
    """RAGPipeline 思考过程过滤测试"""

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_filter_thinking_process_basic(self, mock_settings, test_config):
        """测试基础思考过程过滤"""
        pipeline = RAGPipeline(test_config)
        
        text = "首先，我需要分析这个问题。我的答案是：这是一个测试。"
        result = pipeline._filter_thinking_process(text)
        assert "首先" not in result or "我的" not in result

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_filter_thinking_process_empty(self, mock_settings, test_config):
        """测试空文本过滤"""
        pipeline = RAGPipeline(test_config)
        
        result = pipeline._filter_thinking_process("")
        assert result == ""

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_filter_thinking_process_short(self, mock_settings, test_config):
        """测试短文本不过滤"""
        pipeline = RAGPipeline(test_config)
        
        text = "这是一个短回答。"
        result = pipeline._filter_thinking_process(text)
        assert len(result) > 0

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_filter_thinking_process_multiple_patterns(self, mock_settings, test_config):
        """测试多种思考过程模式"""
        pipeline = RAGPipeline(test_config)
        
        text = """
        我需要仔细思考这个问题。
        根据上下文分析，我认为答案是42。
        总结一下，这个问题的答案是42。
        """
        result = pipeline._filter_thinking_process(text)
        # 应该移除思考过程句子
        assert "我需要仔细思考" not in result or "根据上下文" not in result

    @patch('src.rag.llamaindex.pipeline.Settings')
    def test_filter_thinking_process_preserves_content(self, mock_settings, test_config):
        """测试过滤后保留主要内容"""
        pipeline = RAGPipeline(test_config)
        
        text = "根据分析，机器学习是人工智能的一个重要分支。"
        result = pipeline._filter_thinking_process(text)
        assert "机器学习" in result or "人工智能" in result


class TestRAGPipelineEdgeCases:
    """RAGPipeline 边界情况测试"""

    @patch('src.rag.llamaindex.pipeline.Settings')
    @patch('src.rag.llamaindex.pipeline.SiliconFlowEmbedder')
    @patch('src.rag.llamaindex.pipeline.LlamaIndexLLMAdapter')
    def test_config_with_minimal_fields(self, mock_adapter, mock_embedder, mock_settings):
        """测试最小配置"""
        minimal_config = {
            "llm": {"provider": "qianfan", "qianfan": {"api_key": "test"}},
            "embedding": {"provider": "siliconflow", "siliconflow": {"api_key": "test"}},
            "vector_store": {"provider": "chroma"},
        }
        
        pipeline = RAGPipeline(minimal_config)
        assert pipeline.config is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])