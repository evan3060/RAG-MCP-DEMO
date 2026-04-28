# 测试指南

本文档详细介绍 RAG-MCP-DEMO 项目的测试策略、测试用例编写规范以及如何运行测试。

## 目录

1. [测试概述](#1-测试概述)
2. [测试结构](#2-测试结构)
3. [运行测试](#3-运行测试)
4. [编写测试](#4-编写测试)
5. [测试覆盖率](#5-测试覆盖率)
6. [最佳实践](#6-最佳实践)

---

## 1. 测试概述

### 1.1 测试策略

本项目采用多层次测试策略：

| 测试类型 | 说明 | 运行频率 |
|----------|------|----------|
| 单元测试 | 测试单个函数/类 | 每次提交 |
| 集成测试 | 测试模块间交互 | 每次提交 |
| 端到端测试 | 测试完整流程 | 每日构建 |
| 性能测试 | 测试性能和资源使用 | 每周 |

### 1.2 测试框架

- **pytest**：主测试框架
- **pytest-asyncio**：异步测试支持
- **pytest-mock**：Mock 对象支持

### 1.3 测试标记

项目使用 pytest 标记来分类测试：

```python
@pytest.mark.unit       # 单元测试
@pytest.mark.integration  # 集成测试
@pytest.mark.slow       # 耗时测试
@pytest.mark.requires_api  # 需要 API 密钥的测试
```

---

## 2. 测试结构

### 2.1 目录结构

```
tests/
├── __init__.py
├── conftest.py              # pytest 配置和共享 fixtures
├── unit/                    # 单元测试
│   ├── __init__.py
│   ├── test_config.py       # 配置模块测试
│   ├── test_pipeline.py     # Pipeline 测试
│   ├── test_smart_text_processor.py  # 文本处理测试
│   ├── test_hybrid_retriever.py      # 混合检索测试
│   ├── test_components.py  # 组件基类测试
│   └── test_mcp_server.py  # MCP 服务器测试
└── integration/            # 集成测试
    ├── __init__.py
    ├── test_mcp_server.py  # MCP 服务器集成测试
    └── test_full_pipeline.py  # 完整流程测试
```

### 2.2 conftest.py 提供的 Fixtures

| Fixture | 说明 |
|---------|------|
| `sample_config` | 示例配置字典 |
| `mock_env_vars` | 设置测试环境变量 |
| `temp_knowledge_base` | 创建临时知识库目录 |
| `mock_llm` | 模拟 LLM 对象 |
| `mock_embedder` | 模拟嵌入模型 |
| `mock_vector_store` | 模拟向量存储 |
| `sample_text_nodes` | 示例文本节点 |
| `sample_documents` | 示例文档列表 |
| `sample_search_results` | 示例搜索结果 |

---

## 3. 运行测试

### 3.1 运行所有测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行测试并显示详细输出
pytest tests/ -v --tb=long
```

### 3.2 运行特定类型的测试

```bash
# 仅运行单元测试
pytest tests/unit/ -v

# 仅运行集成测试
pytest tests/integration/ -v

# 排除慢速测试
pytest tests/ -v -m "not slow"
```

### 3.3 运行特定模块的测试

```bash
# 运行配置模块测试
pytest tests/unit/test_config.py -v

# 运行 Pipeline 测试
pytest tests/unit/test_pipeline.py -v

# 运行文本处理测试
pytest tests/unit/test_smart_text_processor.py -v

# 运行混合检索测试
pytest tests/unit/test_hybrid_retriever.py -v
```

### 3.4 运行特定测试

```bash
# 运行单个测试函数
pytest tests/unit/test_config.py::test_load_config_from_file -v

# 运行包含特定关键字的测试
pytest tests/ -v -k "test_ask"
```

### 3.5 生成测试覆盖率报告

```bash
# 安装 coverage 插件
pip install pytest-cov

# 运行测试并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

---

## 4. 编写测试

### 4.1 单元测试示例

```python
"""SmartTextProcessor 单元测试"""

import pytest
from src.rag.llamaindex.pipeline import SmartTextProcessor


class TestSmartTextProcessor:
    """SmartTextProcessor 测试类"""

    def test_initialization_default(self):
        """测试默认初始化"""
        processor = SmartTextProcessor()
        assert processor.doc_type == 'auto'
        assert processor.chunk_size == (200, 500)

    def test_basic_clean_remove_control_chars(self):
        """测试移除控制字符"""
        processor = SmartTextProcessor()
        text = "Hello\x00World\x07Test\n"
        result = processor._basic_clean(text)
        assert '\x00' not in result
        assert '\x07' not in result

    # ... 更多测试
```

### 4.2 异步测试示例

```python
"""异步测试示例"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestHybridRetriever:
    """HybridRetriever 测试类"""

    @pytest.mark.asyncio
    async def test_aretrieve_combined(self, mock_index, mock_vector_retriever):
        """测试组合检索"""
        # 模拟 BM25 检索器
        mock_bm25 = MagicMock()
        mock_bm25.aretrieve = AsyncMock(return_value=[
            NodeWithScore(node=nodes[0], score=0.9),
        ])

        with patch.object(HybridRetriever, '_get_bm25_retriever', return_value=mock_bm25):
            retriever = HybridRetriever(...)
            results = await retriever.aretrieve("test query")
            
            assert len(results) <= 10
```

### 4.3 集成测试示例

```python
"""集成测试示例"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline():
    """测试完整流程"""
    # 准备测试数据
    temp_kb = tmp_path / "knowledge_base"
    temp_kb.mkdir()
    (temp_kb / "test.txt").write_text("测试内容")
    
    # 创建 pipeline
    pipeline = RAGPipeline(config)
    
    # 构建索引
    index = await pipeline.build_index(str(temp_kb))
    assert index is not None
    
    # 问答
    result = await pipeline.ask("什么是测试？")
    assert "answer" in result
    assert "sources" in result
```

### 4.4 Mock 使用示例

```python
"""Mock 使用示例"""

from unittest.mock import MagicMock, AsyncMock, patch


def test_with_mock_llm():
    """使用 mock LLM 测试"""
    # 创建 mock
    mock_llm = MagicMock()
    mock_llm.chat = MagicMock(return_value="模拟回答")
    mock_llm.achat = AsyncMock(return_value="异步模拟回答")
    
    # 使用 mock
    with patch('src.rag.llamaindex.pipeline.Settings') as mock_settings:
        mock_settings.llm = mock_llm
        
        # 执行测试
        pipeline = RAGPipeline(config)
        # ... 断言
```

---

## 5. 测试覆盖率

### 5.1 当前覆盖率目标

| 模块 | 目标覆盖率 |
|------|------------|
| `src/utils/` | 90% |
| `src/rag/llamaindex/` | 80% |
| `src/rag/components/` | 70% |
| `src/mcp_server/` | 85% |

### 5.2 运行覆盖率检查

```bash
# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing

# 生成 HTML 报告
pytest tests/ --cov=src --cov-report=html --cov-report=xml
```

---

## 6. 最佳实践

### 6.1 测试命名规范

- 测试类：`Test<类名>` 或 `Test<功能名>`
- 测试方法：`test_<功能描述>`

```python
class TestSmartTextProcessor:
    def test_initialization_default(self):
        ...
    
    def test_process_returns_nodes(self):
        ...
```

### 6.2 测试结构

每个测试方法应该：
1. **准备**：设置测试数据和 mocks
2. **执行**：调用被测试的函数/方法
3. **断言**：验证结果

```python
def test_something(self):
    # 1. 准备
    processor = SmartTextProcessor()
    text = "测试文本"
    
    # 2. 执行
    nodes = processor.process(text)
    
    # 3. 断言
    assert len(nodes) > 0
    assert nodes[0].text is not None
```

### 6.3 避免测试依赖

- 每个测试应该是独立的
- 不要在测试之间共享状态
- 使用 fixtures 准备测试数据

### 6.4 边界情况测试

必须测试的边界情况：

| 边界情况 | 示例 |
|----------|------|
| 空输入 | 空字符串、空列表 |
| 最小值 | top_k=1 |
| 最大值 | top_k=1000 |
| 特殊字符 | emoji、表情符号 |
| None 值 | None 参数 |
| 异常情况 | API 超时、网络错误 |

### 6.5 测试数据

- 使用有意义的测试数据
- 避免硬编码敏感信息
- 使用 fixtures 复用测试数据

---

## 测试命令速查表

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行特定文件
pytest tests/unit/test_pipeline.py -v

# 运行特定测试函数
pytest tests/unit/test_pipeline.py::test_ask_without_index -v

# 运行包含关键字的测试
pytest tests/ -v -k "test_ask"

# 排除特定标记的测试
pytest tests/ -v -m "not slow"

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing

# 并行运行测试（需要安装 pytest-xdist）
pytest tests/ -n auto

# 显示测试执行的详细信息
pytest tests/ -v -s --tb=long
```

---

## 下一步

- 想要运行测试？查看上面的命令
- 想要添加新测试？遵循本文档的规范
- 想要提高测试覆盖率？运行覆盖率检查并补充测试