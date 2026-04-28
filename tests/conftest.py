"""pytest 配置和共享 fixtures"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "llm": {
            "provider": "qianfan",
            "qianfan": {
                "api_key": "test-api-key",
                "model": "ERNIE-Bot-4",
                "base_url": "https://qianfan.baidubce.com/v2/coding"
            }
        },
        "embedding": {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key": "test-embedding-key",
                "model": "BAAI/bge-large-zh-v1.5"
            }
        },
        "reranker": {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key": "test-reranker-key",
                "model": "BAAI/bge-reranker-v2-m3"
            }
        },
        "vector_store": {
            "provider": "chroma",
            "persist_dir": "./data/chroma_db"
        }
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("LLM_PROVIDER", "qianfan")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "ERNIE-Bot-4")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "siliconflow")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_KEY", "test-reranker-key")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "chroma")
    monkeypatch.setenv("VECTOR_DB_PERSIST_DIR", "./data/chroma_db")


@pytest.fixture
def temp_knowledge_base(tmp_path):
    """创建临时知识库目录"""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    # 创建测试文档
    (kb_dir / "test1.txt").write_text("这是第一个测试文档。\n\n包含多段文字。", encoding="utf-8")
    (kb_dir / "test2.md").write_text("# 测试Markdown\n\n## 第二章\n\n这是第二段内容。", encoding="utf-8")

    return kb_dir


@pytest.fixture
def mock_llm():
    """模拟 LLM"""
    llm = MagicMock()
    llm.chat = MagicMock(return_value="这是一个模拟的回答。")
    llm.achat = AsyncMock(return_value="这是一个模拟的回答。")
    llm.stream_chat = MagicMock(return_value=iter(["这是", "模拟", "回答"]))
    return llm


@pytest.fixture
def mock_embedder():
    """模拟嵌入模型"""
    embedder = MagicMock()
    embedder.get_embedding = MagicMock(return_value=[0.1] * 1024)
    embedder.get_embeddings = MagicMock(return_value=[[0.1] * 1024] * 5)
    return embedder


@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    store = MagicMock()
    store.add = MagicMock()
    store.query = MagicMock(return_value=[])
    store.delete = MagicMock()
    return store


@pytest.fixture
def sample_text_nodes():
    """示例文本节点"""
    from llama_index.core.schema import TextNode

    return [
        TextNode(text="第一段内容关于机器学习。", metadata={"file_name": "test1.txt"}),
        TextNode(text="第二段内容关于深度学习。", metadata={"file_name": "test1.txt"}),
        TextNode(text="第三段内容关于自然语言处理。", metadata={"file_name": "test2.txt"}),
    ]


@pytest.fixture
def sample_documents():
    """示例文档列表"""
    from llama_index.core import Document

    return [
        Document(text="这是第一个文档的内容。", metadata={"file_name": "doc1.txt"}),
        Document(text="这是第二个文档的内容。", metadata={"file_name": "doc2.txt"}),
    ]


@pytest.fixture
def sample_search_results():
    """示例搜索结果"""
    from llama_index.core.schema import TextNode, NodeWithScore

    nodes = [
        TextNode(text="机器学习是人工智能的一个分支。", metadata={"file_name": "ml.txt"}),
        TextNode(text="深度学习是机器学习的子领域。", metadata={"file_name": "dl.txt"}),
        TextNode(text="神经网络是深度学习的基础。", metadata={"file_name": "nn.txt"}),
    ]

    return [
        NodeWithScore(node=node, score=0.95 - i * 0.1)
        for i, node in enumerate(nodes)
    ]


# pytest 标记
def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 耗时测试")
    config.addinivalue_line("markers", "requires_api: 需要 API 密钥的测试")