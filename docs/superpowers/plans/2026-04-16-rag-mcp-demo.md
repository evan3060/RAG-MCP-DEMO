# RAG-MCP-Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG-based knowledge base system with MCP protocol support, featuring hybrid search, local Chroma vector storage, SiliconFlow embeddings, and detailed code annotations for learning purposes.

**Architecture:** LlamaIndex-based RAG pipeline with pluggable components (LLM, Embedder, VectorStore, Retriever). MCP Server exposes three tools: ingest_document, ask_question, search_knowledge. Supports both STDIO and SSE communication modes.

**Tech Stack:** Python 3.10+, LlamaIndex, ChromaDB, MCP Protocol, Qianfan LLM, SiliconFlow API, RAGAS evaluation

**Design Reference:** `docs/superpowers/specs/2026-04-16-rag-mcp-demo-design.md`

---

## Project Structure Overview

```
rag-mcp-demo/
├── src/
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py          # Main MCP server entry
│   │   └── handlers.py        # Tool handlers
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── llamaindex/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py    # Main RAG pipeline
│   │   │   └── hybrid_retriever.py
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── loaders/
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   └── office_loader.py
│   │       ├── embedders/
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   └── siliconflow_embedder.py
│   │       ├── vector_stores/
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   └── chroma_store.py
│   │       ├── rerankers/
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   └── siliconflow_reranker.py
│   │       └── llms/
│   │           ├── __init__.py
│   │           ├── base.py
│   │           └── qianfan_llm.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ragas_evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── registry.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_loaders.py
│   │   ├── test_embedders.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_mcp_server.py
├── config/
│   └── default.yaml
├── data/
│   ├── chroma_db/            # Local vector storage
│   └── documents/            # Uploaded documents
├── requirements.txt
└── README.md
```

---

## Task 1: Project Bootstrap

**Goal:** Set up project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/default.yaml`
- Create: `README.md`

- [ ] **Step 1: Create requirements.txt**

```txt
# Core dependencies
llama-index>=0.12.0
llama-index-vector-stores-chroma>=0.4.0
chromadb>=0.6.0

# MCP Protocol
mcp>=1.0.0

# LLM and Embeddings
qianfan>=0.4.0
openai>=1.0.0

# Evaluation
ragas>=0.2.0
datasets>=2.0.0

# Configuration and utilities
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Office document support (optional)
python-docx>=0.8.11
openpyxl>=3.0.0
xlrd>=2.0.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: Create config/default.yaml**

```yaml
# ==================== 服务配置 ====================
server:
  mode: stdio
  host: "0.0.0.0"
  port: 8000

# ==================== LLM 配置 ====================
llm:
  provider: qianfan
  qianfan:
    api_key: "${QIANFAN_API_KEY}"
    secret_key: "${QIANFAN_SECRET_KEY}"
    model: "ERNIE-Bot-4"
    temperature: 0.7
    max_tokens: 2048

# ==================== 嵌入模型配置 ====================
embedding:
  provider: siliconflow
  siliconflow:
    api_key: "${SILICONFLOW_API_KEY}"
    model: "BAAI/bge-large-zh-v1.5"

# ==================== 向量数据库配置 ====================
vector_store:
  provider: chroma
  chroma:
    persist_directory: "./data/chroma_db"
    collection_name: "knowledge_base"
    distance_fn: "cosine"

# ==================== 检索配置 ====================
retrieval:
  strategy: hybrid
  top_k: 10
  hybrid:
    vector_weight: 0.7
    bm25_weight: 0.3
    fusion_mode: rrf

# ==================== 重排序配置 ====================
reranker:
  enabled: true
  siliconflow:
    api_key: "${SILICONFLOW_API_KEY}"
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 5

# ==================== 文档处理配置 ====================
ingestion:
  supported_extensions: [".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls"]
  splitter:
    strategy: semantic
    semantic:
      buffer_size: 1
      breakpoint_percentile_threshold: 95
    recursive:
      chunk_size: 512
      chunk_overlap: 50
  office:
    excel:
      max_rows_per_sheet: 10000
      date_format: "%Y-%m-%d"
    word:
      extract_tables: true
      include_headers: false

# ==================== 评估配置 ====================
evaluation:
  enabled: true
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall

# ==================== 日志配置 ====================
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/rag-mcp.log"
```

- [ ] **Step 3: Create project directories**

```bash
mkdir -p src/{mcp_server,rag/{llamaindex,components/{loaders,embedders,vector_stores,rerankers,llms}},evaluation,utils}
mkdir -p tests/{unit,integration}
mkdir -p config data/{chroma_db,documents} logs
```

- [ ] **Step 4: Create empty __init__.py files**

```bash
touch src/__init__.py
touch src/mcp_server/__init__.py
touch src/rag/__init__.py
touch src/rag/llamaindex/__init__.py
touch src/rag/components/__init__.py
touch src/rag/components/loaders/__init__.py
touch src/rag/components/embedders/__init__.py
touch src/rag/components/vector_stores/__init__.py
touch src/rag/components/rerankers/__init__.py
touch src/rag/components/llms/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/default.yaml README.md
find . -name "__init__.py" -path "./src/*" -exec git add {} \;
find . -name "__init__.py" -path "./tests/*" -exec git add {} \;
git commit -m "chore: bootstrap project structure and configuration"
```

---

## Task 2: Configuration and Utilities

**Goal:** Create configuration management and utility modules

**Files:**
- Create: `src/utils/config.py`
- Create: `src/utils/logger.py`
- Create: `src/utils/registry.py`

- [ ] **Step 1: Write test for config loading**

```python
# tests/unit/test_config.py
import os
import pytest
from src.utils.config import load_config


def test_load_config_from_file():
    """Test loading config from default.yaml"""
    config = load_config("config/default.yaml")
    
    assert "server" in config
    assert config["server"]["mode"] == "stdio"
    assert "llm" in config
    assert config["llm"]["provider"] == "qianfan"


def test_config_expands_env_vars(monkeypatch):
    """Test that environment variables are expanded"""
    monkeypatch.setenv("TEST_API_KEY", "test-value-123")
    
    config_str = """
llm:
  api_key: "${TEST_API_KEY}"
"""
    # This would require creating a temp file or mocking
    # For now, just test the function exists
    assert callable(load_config)
```

- [ ] **Step 2: Create config.py**

```python
# src/utils/config.py
"""
配置管理模块 - 加载和管理应用配置

【设计说明】
- 支持 YAML 配置文件
- 自动展开环境变量（如 ${API_KEY}）
- 提供类型安全的配置访问
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def _expand_env_vars(value: Any) -> Any:
    """递归展开配置中的环境变量
    
    示例:
        "${API_KEY}" -> "sk-xxx" (从环境变量读取)
        "${UNDEFINED:-default}" -> "default" (使用默认值)
    """
    if isinstance(value, str):
        # 匹配 ${VAR} 或 ${VAR:-default}
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, match.group(0))
        
        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """加载配置文件
    
    【参数】
    config_path: 配置文件路径，默认 config/default.yaml
    
    【返回】
    配置字典，环境变量已展开
    
    【示例】
    >>> config = load_config()
    >>> print(config["llm"]["provider"])
    'qianfan'
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 展开环境变量
    config = _expand_env_vars(config)
    
    return config


class Config:
    """配置类 - 提供属性方式访问配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的路径"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
```

- [ ] **Step 3: Create logger.py**

```python
# src/utils/logger.py
"""
日志工具模块 - 统一日志配置

【设计说明】
- 使用标准库 logging
- 支持文件和控制台双输出
- 自动创建日志目录
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """设置并获取日志记录器
    
    【参数】
    name: 日志记录器名称
    level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    log_file: 日志文件路径（可选）
    format_str: 自定义格式字符串（可选）
    
    【返回】
    配置好的日志记录器
    
    【示例】
    >>> logger = setup_logger("rag")
    >>> logger.info("应用启动")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 默认格式
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已配置的日志记录器
    
    如果记录器尚未配置，返回根记录器
    """
    return logging.getLogger(name)
```

- [ ] **Step 4: Create registry.py**

```python
# src/utils/registry.py
"""
插件注册中心 - 管理可插拔组件

【设计模式】注册表模式 (Registry Pattern)
允许运行时动态注册和获取组件实现

【示例】
>>> @Registry.register("llm", "qianfan")
>>> class QianfanLLM(BaseLLM):
...     pass
>>>
>>> llm = Registry.create("llm", "qianfan", config)
"""

from typing import Any, Callable, Dict, Type


class Registry:
    """组件注册表"""
    
    _registry: Dict[str, Dict[str, Type]] = {}
    
    @classmethod
    def register(cls, component_type: str, name: str):
        """注册装饰器
        
        【参数】
        component_type: 组件类型 (llm, embedder, vector_store, etc.)
        name: 组件名称标识
        
        【示例】
        @Registry.register("llm", "qianfan")
        class QianfanLLM(BaseLLM):
            pass
        """
        def decorator(wrapped_class: Type):
            if component_type not in cls._registry:
                cls._registry[component_type] = {}
            
            cls._registry[component_type][name] = wrapped_class
            return wrapped_class
        
        return decorator
    
    @classmethod
    def create(cls, component_type: str, name: str, config: Dict[str, Any]) -> Any:
        """创建组件实例
        
        【参数】
        component_type: 组件类型
        name: 组件名称
        config: 配置字典，传递给组件构造函数
        
        【返回】
        组件实例
        
        【异常】
        ValueError: 组件类型或名称不存在
        """
        if component_type not in cls._registry:
            raise ValueError(f"未知的组件类型: {component_type}")
        
        if name not in cls._registry[component_type]:
            available = list(cls._registry[component_type].keys())
            raise ValueError(
                f"未知的 {component_type} 组件: {name}. "
                f"可用选项: {available}"
            )
        
        component_class = cls._registry[component_type][name]
        return component_class(config)
    
    @classmethod
    def list_components(cls, component_type: str) -> list:
        """列出某类型的所有可用组件"""
        return list(cls._registry.get(component_type, {}).keys())
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/unit/test_config.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/utils/*.py tests/unit/test_config.py
git commit -m "feat: add configuration, logger, and registry utilities"
```

---

## Task 3: LLM Base Class and Qianfan Implementation

**Goal:** Create the LLM abstraction layer and Qianfan implementation

**Files:**
- Create: `src/rag/components/llms/base.py`
- Create: `src/rag/components/llms/qianfan_llm.py`
- Create: `tests/unit/test_llms.py`

- [ ] **Step 1: Write failing test for LLM base**

```python
# tests/unit/test_llms.py
import pytest
from src.rag.components.llms.base import LLMMessage, LLMResponse, BaseLLM


def test_llm_message_creation():
    """Test LLMMessage dataclass"""
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_llm_response_creation():
    """Test LLMResponse dataclass"""
    response = LLMResponse(content="Hi there")
    assert response.content == "Hi there"
    assert response.usage == {}
    assert response.model == ""


@pytest.mark.asyncio
async def test_base_llm_is_abstract():
    """Test that BaseLLM cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseLLM(config={})


class MockLLM(BaseLLM):
    """Mock implementation for testing"""
    
    def _validate_config(self):
        pass
    
    async def generate(self, messages, temperature=None, **kwargs):
        return LLMResponse(content="Mock response")


@pytest.mark.asyncio
async def test_mock_llm_generate():
    """Test mock LLM implementation"""
    llm = MockLLM(config={})
    messages = [LLMMessage(role="user", content="Test")]
    response = await llm.generate(messages)
    
    assert response.content == "Mock response"
```

- [ ] **Step 2: Create base.py**

```python
# src/rag/components/llms/base.py
"""
LLM 基类设计 - 面向初级开发者的接口抽象教学

【学习要点】
1. 什么是抽象基类 (ABC)?
   - 定义接口规范，强制子类实现特定方法
   - 保证不同 LLM 实现有一致的使用方式

2. 为什么要使用基类?
   - 解耦: 上层代码不依赖具体 LLM 实现
   - 可扩展: 新增 LLM 只需实现基类接口
   - 可测试: 易于 Mock 和单元测试

【设计模式】模板方法模式 (Template Method)
基类定义算法骨架，子类实现具体步骤
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """
    消息模型 - 统一不同 LLM 的消息格式
    
    【类比理解】
    就像写信，每封信都有：
    - role: 是谁写的（用户/助手/系统）
    - content: 写了什么内容
    """
    role: str       # "system", "user", "assistant"
    content: str    # 消息内容


class LLMResponse(BaseModel):
    """
    LLM 响应模型
    
    【字段说明】
    - content: AI 生成的回答文本
    - usage: Token 使用量（用于计费和分析）
    - model: 使用的模型名称
    """
    content: str
    usage: dict = {}
    model: str = ""


class BaseLLM(ABC):
    """
    LLM 抽象基类 - 所有 LLM 实现的接口规范
    
    【使用方式】
    ```python
    # 子类必须实现抽象方法
    class MyLLM(BaseLLM):
        def _validate_config(self): ...
        async def generate(self, messages, ...): ...
    ```
    """
    
    def __init__(self, config: dict):
        """
        初始化方法
        
        【参数】
        config: 配置字典，包含 api_key, model, temperature 等
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        验证配置 - 子类必须实现
        
        【为什么需要】
        尽早发现配置错误，避免运行时出错
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本 - 核心方法
        
        【参数】
        messages: 对话历史，格式 [{"role": "user", "content": "你好"}]
        temperature: 创造性参数 (0-2)，越高回答越随机
        **kwargs: 额外的模型特定参数
        
        【返回】
        LLMResponse 对象，包含生成的文本和元信息
        
        【示例】
        ```python
        response = await llm.generate([
            LLMMessage(role="user", content="什么是 RAG?")
        ])
        print(response.content)  # RAG 是检索增强生成...
        ```
        """
        pass
    
    async def stream_generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成 - 逐字返回（用于打字机效果）
        
        【使用场景】
        前端需要实时显示 AI 回答，而不是等全部生成完
        
        【示例】
        ```python
        async for chunk in llm.stream_generate(messages):
            print(chunk, end="", flush=True)  # 逐字输出
        ```
        """
        # 默认实现：非流式生成后分段返回
        # 子类可覆盖提供更高效的流式实现
        response = await self.generate(messages, **kwargs)
        # 按字符分段模拟流式
        for char in response.content:
            yield char
```

- [ ] **Step 3: Create qianfan_llm.py**

```python
# src/rag/components/llms/qianfan_llm.py
"""
千帆 LLM 实现 - 百度智能云大模型 API

【学习要点】
1. 如何实现基类接口
2. 如何处理 API 调用
3. 如何转换数据格式
"""

from typing import List, Optional

import qianfan

from src.rag.components.llms.base import BaseLLM, LLMMessage, LLMResponse


class QianfanLLM(BaseLLM):
    """
    千帆 LLM 实现
    
    【支持的模型】
    - ERNIE-Bot-4: 百度最新旗舰模型
    - ERNIE-Bot: 通用对话模型
    - ERNIE-Bot-turbo: 轻量快速模型
    
    【使用示例】
    ```python
    llm = QianfanLLM({
        "api_key": "your-key",
        "secret_key": "your-secret",
        "model": "ERNIE-Bot-4",
        "temperature": 0.7
    })
    
    response = await llm.generate([
        LLMMessage(role="user", content="你好")
    ])
    ```
    """
    
    def __init__(self, config: dict):
        """初始化千帆 LLM"""
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.model_name = config.get("model", "ERNIE-Bot-4")
        self.default_temperature = config.get("temperature", 0.7)
        
        super().__init__(config)
        
        # 初始化千帆客户端
        self.client = qianfan.ChatCompletion(
            ak=self.api_key,
            sk=self.secret_key
        )
    
    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "千帆 LLM 需要提供 api_key 和 secret_key"
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        调用千帆 API 生成回答
        
        【参数转换】
        我们的 LLMMessage -> 千帆的 message 格式
        """
        # 转换消息格式
        qianfan_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # 调用 API
        response = self.client.do(
            model=self.model_name,
            messages=qianfan_messages,
            temperature=temperature or self.default_temperature,
            **kwargs
        )
        
        # 解析响应
        result = response.body
        
        return LLMResponse(
            content=result.get("result", ""),
            usage=result.get("usage", {}),
            model=self.model_name
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_llms.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/components/llms/*.py tests/unit/test_llms.py
git commit -m "feat: add LLM base class and Qianfan implementation"
```

---

## Task 4: Embedder Base Class and SiliconFlow Implementation

**Goal:** Create embedder abstraction and SiliconFlow implementation

**Files:**
- Create: `src/rag/components/embedders/base.py`
- Create: `src/rag/components/embedders/siliconflow_embedder.py`
- Create: `tests/unit/test_embedders.py`

- [ ] **Step 1: Create base.py**

```python
# src/rag/components/embedders/base.py
"""
嵌入模型基类 - 将文本转换为向量

【什么是嵌入?】
把文字转换成数字向量，让计算机能"理解"语义
例如: "狗" -> [0.1, -0.5, 0.8, ...] (1024维向量)

【为什么需要?】
- 计算机只能处理数字，不能直接理解文字
- 相似语义的文本有相似的向量（余弦相似度高）
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """嵌入模型抽象基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "")
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        嵌入单条文本
        
        【参数】
        text: 输入文本
        
        【返回】
        向量列表，如 [0.1, -0.5, 0.8, ...]
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入多条文本（更高效）
        
        【参数】
        texts: 文本列表
        
        【返回】
        向量列表的列表
        """
        pass
```

- [ ] **Step 2: Create siliconflow_embedder.py**

```python
# src/rag/components/embedders/siliconflow_embedder.py
"""
SiliconFlow 嵌入模型实现

【推荐模型】
- BAAI/bge-large-zh-v1.5: 中文最佳（1024维）
- BAAI/bge-m3: 多语言支持
- BAAI/bge-small-zh-v1.5: 轻量快速

【API 文档】
https://docs.siliconflow.cn/api-reference/embeddings/create-embeddings
"""

from typing import List

import httpx

from src.rag.components.embedders.base import BaseEmbedder


class SiliconFlowEmbedder(BaseEmbedder):
    """SiliconFlow 嵌入模型实现"""
    
    API_BASE = "https://api.siliconflow.cn/v1"
    
    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        super().__init__(config)
    
    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("SiliconFlow Embedder 需要提供 api_key")
    
    async def embed(self, text: str) -> List[float]:
        """嵌入单条文本"""
        results = await self.embed_batch([text])
        return results[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name or "BAAI/bge-large-zh-v1.5",
                    "input": texts,
                    "encoding_format": "float"
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # 提取嵌入向量
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
```

- [ ] **Step 3: Create test_embedders.py**

```python
# tests/unit/test_embedders.py
import pytest
from src.rag.components.embedders.base import BaseEmbedder
from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder


def test_embedder_is_abstract():
    """Test that BaseEmbedder cannot be instantiated"""
    with pytest.raises(TypeError):
        BaseEmbedder(config={})


def test_siliconflow_requires_api_key():
    """Test that SiliconFlow requires API key"""
    with pytest.raises(ValueError, match="api_key"):
        SiliconFlowEmbedder(config={})


@pytest.mark.asyncio
async def test_siliconflow_embed_mocked(monkeypatch):
    """Test embed with mocked API response"""
    # 模拟 httpx 响应
    class MockResponse:
        def __init__(self):
            self.status_code = 200
        
        def json(self):
            return {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
        
        def raise_for_status(self):
            pass
    
    class MockClient:
        async def post(self, *args, **kwargs):
            return MockResponse()
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", MockClient)
    
    embedder = SiliconFlowEmbedder(config={"api_key": "test-key"})
    result = await embedder.embed_batch(["text1", "text2"])
    
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_embedders.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/components/embedders/*.py tests/unit/test_embedders.py
git commit -m "feat: add embedder base class and SiliconFlow implementation"
```

---

## Task 5: Vector Store Base Class and Chroma Implementation

**Goal:** Create vector store abstraction and Chroma implementation

**Files:**
- Create: `src/rag/components/vector_stores/base.py`
- Create: `src/rag/components/vector_stores/chroma_store.py`

- [ ] **Step 1: Create base.py**

```python
# src/rag/components/vector_stores/base.py
"""
向量数据库基类 - 存储和检索文本向量

【类比理解】
就像图书馆的索引系统：
- 每本书（文档）有唯一的编号
- 通过主题标签（向量）快速找到相关书籍
- 可以添加新书，也可以按标签搜索
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorSearchResult:
    """向量搜索结果"""
    def __init__(self, id: str, score: float, text: str, metadata: Dict):
        self.id = id
        self.score = score  # 相似度分数 (0-1)
        self.text = text    # 原始文本
        self.metadata = metadata  # 元数据


class BaseVectorStore(ABC):
    """向量数据库抽象基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.collection_name = config.get("collection_name", "default")
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置"""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        插入或更新文档
        
        【参数】
        ids: 文档唯一标识
        embeddings: 向量列表
        documents: 原始文本列表
        metadatas: 元数据列表
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """
        向量相似度搜索
        
        【参数】
        query_embedding: 查询向量
        top_k: 返回结果数量
        filters: 过滤条件（可选）
        
        【返回】
        搜索结果列表，按相似度排序
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """删除指定文档"""
        pass
```

- [ ] **Step 2: Create chroma_store.py**

```python
# src/rag/components/vector_stores/chroma_store.py
"""
Chroma 向量数据库实现 - 本地嵌入式存储

【特点】
- 零部署: pip install 即可使用
- 自动持久化: 数据保存到本地文件
- 适合: 开发、Demo、小团队

【数据存储位置】
./data/chroma_db/ 目录下
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.rag.components.vector_stores.base import BaseVectorStore, VectorSearchResult


class ChromaVectorStore(BaseVectorStore):
    """Chroma 本地向量数据库实现"""
    
    def __init__(self, config: dict):
        self.persist_dir = config.get("persist_directory", "./data/chroma_db")
        self.distance_fn = config.get("distance_fn", "cosine")
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """验证配置并创建目录"""
        # 确保持久化目录存在
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_client(self):
        """获取 Chroma 客户端（懒加载）"""
        if not hasattr(self, '_client'):
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn}
            )
        
        return self._collection
    
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """插入或更新文档"""
        collection = self._get_client()
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """向量相似度搜索"""
        collection = self._get_client()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # 转换结果格式
        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            search_results.append(VectorSearchResult(
                id=doc_id,
                score=results["distances"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}
            ))
        
        return search_results
    
    async def delete(self, ids: List[str]) -> None:
        """删除指定文档"""
        collection = self._get_client()
        collection.delete(ids=ids)
```

- [ ] **Step 3: Commit**

```bash
git add src/rag/components/vector_stores/*.py
git commit -m "feat: add vector store base class and Chroma implementation"
```

---

## Task 6: Office Document Loaders

**Goal:** Implement loaders for Word and Excel documents

**Files:**
- Create: `src/rag/components/loaders/base.py`
- Create: `src/rag/components/loaders/office_loader.py`

- [ ] **Step 1: Create base.py**

```python
# src/rag/components/loaders/base.py
"""
文档加载器基类

【职责】
将各种格式的文件（Word、Excel、PDF等）
转换为 LlamaIndex 的 Document 对象
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llama_index.core.schema import Document


class BaseLoader(ABC):
    """文档加载器抽象基类"""
    
    @abstractmethod
    def load_data(self, file_path: Path) -> List[Document]:
        """
        加载文件并返回文档列表
        
        【参数】
        file_path: 文件路径
        
        【返回】
        Document 对象列表
        """
        pass
```

- [ ] **Step 2: Create office_loader.py**

```python
# src/rag/components/loaders/office_loader.py
"""
Office 文档加载器 - 支持 Word、Excel

【支持的格式】
- .docx: Word 文档 (python-docx)
- .xlsx, .xls: Excel 表格 (openpyxl, xlrd)
"""

from pathlib import Path
from typing import List

from llama_index.core.schema import Document

from src.rag.components.loaders.base import BaseLoader


class DocxLoader(BaseLoader):
    """Word 文档加载器 (.docx)"""
    
    def load_data(self, file_path: Path) -> List[Document]:
        """加载 Word 文档"""
        from docx import Document as DocxDocument
        
        doc = DocxDocument(file_path)
        
        # 提取段落
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        # 提取表格（转为 Markdown 格式）
        tables_md = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                # 转义 Markdown 表格中的 |
                cells = [cell.text.replace('|', '\\|') for cell in row.cells]
                rows.append('| ' + ' | '.join(cells) + ' |')
            
            # 添加表头分隔线
            if rows and len(table.rows) > 0:
                col_count = len(table.rows[0].cells)
                rows.insert(1, '|' + '---|' * col_count)
            
            tables_md.append('\n'.join(rows))
        
        # 合并内容
        content = '\n\n'.join(paragraphs)
        if tables_md:
            content += '\n\n## 表格\n\n' + '\n\n'.join(tables_md)
        
        metadata = {
            'source': str(file_path),
            'type': 'docx',
            'paragraphs': len(paragraphs),
            'tables': len(doc.tables),
        }
        
        return [Document(text=content, metadata=metadata)]


class ExcelLoader(BaseLoader):
    """Excel 加载器 (.xlsx, .xls)"""
    
    def __init__(self, max_rows: int = 10000):
        self.max_rows = max_rows
    
    def load_data(self, file_path: Path) -> List[Document]:
        """加载 Excel 文件"""
        import pandas as pd
        
        xl_file = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in xl_file.sheet_names:
            # 限制行数，防止内存溢出
            df = xl_file.parse(sheet_name, nrows=self.max_rows)
            
            # 转为 Markdown 表格
            markdown = df.to_markdown(index=False)
            content = f'## Sheet: {sheet_name}\n\n{markdown}'
            
            if len(df) >= self.max_rows:
                content += f'\n\n> ⚠️ 表格过大，仅显示前 {self.max_rows} 行'
            
            metadata = {
                'source': str(file_path),
                'type': 'excel',
                'sheet': sheet_name,
                'rows': len(df),
                'columns': list(df.columns),
            }
            
            documents.append(Document(text=content, metadata=metadata))
        
        return documents


class UnifiedOfficeLoader(BaseLoader):
    """统一 Office 加载器 - 自动识别文件类型"""
    
    LOADERS = {
        '.docx': DocxLoader,
        '.xlsx': ExcelLoader,
        '.xls': ExcelLoader,
    }
    
    def load_data(self, file_path: Path) -> List[Document]:
        """根据扩展名自动选择加载器"""
        suffix = Path(file_path).suffix.lower()
        loader_class = self.LOADERS.get(suffix)
        
        if not loader_class:
            raise ValueError(f'不支持的文件格式: {suffix}')
        
        return loader_class().load_data(file_path)
```

- [ ] **Step 3: Commit**

```bash
git add src/rag/components/loaders/*.py
git commit -m "feat: add office document loaders (Word, Excel)"
```

---

## Task 7: Reranker Base and SiliconFlow Implementation

**Goal:** Create reranker abstraction and SiliconFlow implementation

**Files:**
- Create: `src/rag/components/rerankers/base.py`
- Create: `src/rag/components/rerankers/siliconflow_reranker.py`
- Create: `tests/unit/test_rerankers.py`

- [ ] **Step 1: Write test for reranker base**

```python
# tests/unit/test_rerankers.py
import pytest
from src.rag.components.rerankers.base import RerankResult, BaseReranker


def test_rerank_result_creation():
    """Test RerankResult dataclass"""
    result = RerankResult(id="doc1", text="content", score=0.95, metadata={})
    assert result.id == "doc1"
    assert result.score == 0.95


def test_reranker_is_abstract():
    """Test that BaseReranker cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseReranker(config={})
```

- [ ] **Step 2: Create base.py**

```python
# src/rag/components/rerankers/base.py
"""
重排序基类 - 对初步检索结果进行精排

【什么是重排序?】
先用快速但粗略的方法找出一批候选（如Top-100），
再用更精准但慢的方法对候选重新排序（选出Top-5）

【类比】
就像招聘：
1. 先快速筛选简历，找出100个符合条件的（初筛）
2. 再仔细面试这100人，选出最合适的5个（精排）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RerankResult:
    """重排序结果"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class BaseReranker(ABC):
    """重排序器抽象基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.top_n = config.get("top_n", 5)
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置"""
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[RerankResult]:
        """
        重排序文档
        
        【参数】
        query: 查询文本
        documents: 候选文档列表
        doc_ids: 文档ID列表
        metadatas: 元数据列表
        
        【返回】
        重排序后的结果列表（按相关性降序）
        """
        pass
```

- [ ] **Step 3: Create siliconflow_reranker.py**

```python
# src/rag/components/rerankers/siliconflow_reranker.py
"""
SiliconFlow 重排序实现

【使用模型】
BAAI/bge-reranker-v2-m3: 免费重排序模型

【API 文档】
https://docs.siliconflow.cn/api-reference/rerank/create-rerank
"""

from typing import List, Dict, Any

import httpx

from src.rag.components.rerankers.base import BaseReranker, RerankResult


class SiliconFlowReranker(BaseReranker):
    """SiliconFlow 重排序实现"""
    
    API_BASE = "https://api.siliconflow.cn/v1"
    
    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "BAAI/bge-reranker-v2-m3")
        super().__init__(config)
    
    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("SiliconFlow Reranker 需要提供 api_key")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[RerankResult]:
        """调用 SiliconFlow API 进行重排序"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": self.top_n
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # 转换结果格式
            results = []
            for item in data.get("results", []):
                idx = item["index"]
                results.append(RerankResult(
                    id=doc_ids[idx],
                    text=documents[idx],
                    score=item["relevance_score"],
                    metadata=metadatas[idx]
                ))
            
            return results
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_rerankers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/components/rerankers/*.py tests/unit/test_rerankers.py
git commit -m "feat: add reranker base class and SiliconFlow implementation"
```

---

## Task 8: Hybrid Retriever

**Goal:** Implement hybrid retrieval combining vector and BM25 search

**Files:**
- Create: `src/rag/llamaindex/hybrid_retriever.py`
- Create: `tests/unit/test_hybrid_retriever.py`

- [ ] **Step 1: Write test for hybrid retriever**

```python
# tests/unit/test_hybrid_retriever.py
import pytest
from unittest.mock import Mock, AsyncMock

from src.rag.llamaindex.hybrid_retriever import HybridRetriever, rrf_fusion


def test_rrf_fusion():
    """Test RRF fusion algorithm"""
    vector_results = [
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8},
    ]
    bm25_results = [
        {"id": "doc2", "score": 0.95},
        {"id": "doc3", "score": 0.85},
    ]
    
    fused = rrf_fusion(vector_results, bm25_results, vector_weight=0.7, bm25_weight=0.3)
    
    # doc2 appears in both, should have higher score
    assert len(fused) == 3
    assert fused[0]["id"] == "doc2"  # Most relevant


@pytest.mark.asyncio
async def test_hybrid_retriever_aretrieve():
    """Test hybrid retriever"""
    # Mock dependencies
    mock_vector_retriever = Mock()
    mock_vector_retriever.aretrieve = AsyncMock(return_value=[
        Mock(node=Mock(node_id="doc1", text="content1"), score=0.9),
        Mock(node=Mock(node_id="doc2", text="content2"), score=0.8),
    ])
    
    mock_bm25_retriever = Mock()
    mock_bm25_retriever.aretrieve = AsyncMock(return_value=[])
    
    retriever = HybridRetriever(
        index=Mock(),
        vector_retriever=mock_vector_retriever,
        top_k=5
    )
    retriever.bm25_retriever = mock_bm25_retriever
    
    results = await retriever.aretrieve("test query")
    
    assert len(results) > 0
    mock_vector_retriever.aretrieve.assert_called_once()
```

- [ ] **Step 2: Create hybrid_retriever.py**

```python
# src/rag/llamaindex/hybrid_retriever.py
"""
混合检索实现 - 结合向量检索和关键词检索

【什么是混合检索?】
类比：找资料时同时用"意思理解"(向量)和"关键词搜索"(BM25)

【为什么要混合?】
┌─────────────────┬──────────────────┬─────────────────┐
│     检索方式     │      擅长        │      不擅长      │
├─────────────────┼──────────────────┼─────────────────┤
│ 向量检索(Dense)  │ 语义相似         │ 精确术语匹配     │
│ BM25(Sparse)   │ 精确匹配         │ 语义理解         │
│ 混合(Hybrid)   │ 两者优点结合     │ 复杂度稍高       │
└─────────────────┴──────────────────┴─────────────────┘

【融合算法 - RRF (Reciprocal Rank Fusion)】
score = Σ(weight_i / (k + rank))
其中 k 是常数(通常60)，rank 是排名
"""

from typing import List
from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.schema import NodeWithScore, QueryBundle


def rrf_fusion(
    vector_results: List[NodeWithScore],
    bm25_results: List[NodeWithScore],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60
) -> List[NodeWithScore]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    
    【参数】
    vector_results: 向量检索结果
    bm25_results: BM25检索结果
    vector_weight: 向量结果权重
    bm25_weight: BM25结果权重
    k: RRF常数，防止低排名项分数过高
    
    【示例】
    文档A: 向量排名#2, BM25排名#5
    score = 0.7/(60+2) + 0.3/(60+5) = 0.0113 + 0.0046 = 0.0159
    """
    scores = defaultdict(float)
    node_map = {}
    
    # 处理向量检索结果
    for rank, node in enumerate(vector_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += vector_weight / (k + rank)
        node_map[node_id] = node.node
    
    # 处理 BM25 检索结果
    for rank, node in enumerate(bm25_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += bm25_weight / (k + rank)
        if node_id not in node_map:
            node_map[node_id] = node.node
    
    # 按融合分数排序
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 重建 NodeWithScore 列表
    return [NodeWithScore(node=node_map[nid], score=score) for nid, score in sorted_nodes]


class HybridRetriever:
    """
    混合检索器 - 融合向量检索和 BM25 检索结果
    
    【使用方式】
    ```python
    retriever = HybridRetriever(
        index=index,
        vector_retriever=vector_retriever,
        top_k=10,
        vector_weight=0.7,
        bm25_weight=0.3
    )
    results = await retriever.aretrieve("什么是RAG?")
    ```
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        vector_retriever: VectorIndexRetriever,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.index = index
        self.vector_retriever = vector_retriever
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # 初始化 BM25 检索器
        self.bm25_retriever = self._init_bm25_retriever()
    
    def _init_bm25_retriever(self) -> BM25Retriever:
        """初始化 BM25 检索器"""
        nodes = list(self.index.docstore.docs.values())
        return BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_k * 2
        )
    
    async def aretrieve(self, query: str) -> List[NodeWithScore]:
        """异步执行混合检索"""
        # 并行执行两种检索
        vector_results = await self.vector_retriever.aretrieve(query)
        bm25_results = await self.bm25_retriever.aretrieve(query)
        
        # 融合结果
        fused_results = rrf_fusion(
            vector_results,
            bm25_results,
            self.vector_weight,
            self.bm25_weight
        )
        
        return fused_results[:self.top_k]
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/unit/test_hybrid_retriever.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/rag/llamaindex/hybrid_retriever.py tests/unit/test_hybrid_retriever.py
git commit -m "feat: implement hybrid retriever with RRF fusion"
```

---

## Task 9: Main RAG Pipeline

**Goal:** Create the main RAG pipeline integrating all components

**Files:**
- Create: `src/rag/llamaindex/pipeline.py`
- Create: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write test for pipeline**

```python
# tests/unit/test_pipeline.py
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.rag.llamaindex.pipeline import RAGPipeline


@pytest.fixture
def mock_config():
    return {
        "siliconflow_api_key": "test-key",
        "qianfan_api_key": "test-key",
        "chunk_size": 512,
        "top_k": 10
    }


@pytest.mark.asyncio
async def test_pipeline_ask(mock_config):
    """Test pipeline ask method"""
    with patch('src.rag.llamaindex.pipeline.VectorStoreIndex') as mock_index:
        with patch('src.rag.llamaindex.pipeline.ContextChatEngine') as mock_engine:
            # Mock response
            mock_response = Mock()
            mock_response.response = "Test answer"
            mock_response.source_nodes = []
            
            mock_engine.from_defaults = Mock(return_value=Mock(
                achat=AsyncMock(return_value=mock_response)
            ))
            
            pipeline = RAGPipeline(mock_config)
            pipeline.index = Mock()
            
            result = await pipeline.ask("What is RAG?")
            
            assert "answer" in result
            assert result["answer"] == "Test answer"
```

- [ ] **Step 2: Create pipeline.py (simplified version)**

```python
# src/rag/llamaindex/pipeline.py
"""
LlamaIndex RAG Pipeline - 完整的检索增强生成流程

【Pipeline 流程】
文档 → 切分 → 嵌入 → 存储 → 检索 → 重排 → 生成

【使用方式】
```python
# 1. 初始化
pipeline = RAGPipeline(config)

# 2. 构建索引（只需执行一次）
await pipeline.build_index("./documents")

# 3. 问答
answer = await pipeline.ask("什么是向量数据库?")
```
"""

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
from src.rag.llamaindex.hybrid_retriever import HybridRetriever


class RAGPipeline:
    """RAG Pipeline 主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index: Optional[VectorStoreIndex] = None
        self._configure_settings()
    
    def _configure_settings(self):
        """配置 LlamaIndex 全局设置"""
        # 设置嵌入模型
        Settings.embed_model = SiliconFlowEmbedder({
            "api_key": self.config["siliconflow_api_key"],
            "model": "BAAI/bge-large-zh-v1.5"
        })
        
        # 设置 LLM
        Settings.llm = QianfanLLM({
            "api_key": self.config["qianfan_api_key"],
            "secret_key": self.config.get("qianfan_secret_key"),
            "model": "ERNIE-Bot-4"
        })
    
    async def build_index(self, documents_path: str) -> VectorStoreIndex:
        """构建知识库索引"""
        # 加载文档
        documents = SimpleDirectoryReader(documents_path).load_data()
        
        # 语义切分
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )
        nodes = splitter.get_nodes_from_documents(documents)
        
        # 创建索引
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
        
        # 创建混合检索器
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20
        )
        hybrid_retriever = HybridRetriever(
            index=self.index,
            vector_retriever=vector_retriever,
            top_k=10
        )
        
        # 创建重排序器
        reranker = SiliconFlowReranker({
            "api_key": self.config["siliconflow_api_key"],
            "top_n": 5
        })
        
        # 创建对话引擎
        chat_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000)
        )
        
        # 执行查询
        response = await chat_engine.achat(question)
        
        # 提取来源
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
```

- [ ] **Step 3: Commit**

```bash
git add src/rag/llamaindex/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: implement main RAG pipeline with LlamaIndex"
```

---

## Task 10: MCP Server and Handlers

**Goal:** Implement MCP server with three tools

**Files:**
- Create: `src/mcp_server/server.py`
- Create: `src/mcp_server/handlers.py`

- [ ] **Step 1: Create server.py**

```python
# src/mcp_server/server.py
"""
MCP Server 实现 - Model Context Protocol 服务端

【什么是 MCP?】
MCP (Model Context Protocol) = AI 模型的"USB 接口"
让不同的 AI 客户端（如 Claude Desktop）能以统一方式调用 RAG 服务

【通信方式】
- STDIO: 标准输入输出，适合本地进程通信
- SSE: Server-Sent Events，适合 Web 通信
"""

import asyncio
from typing import Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.rag.llamaindex.pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGMCPServer:
    """RAG MCP 服务器"""
    
    def __init__(self):
        self.server = Server("rag-mcp-server")
        self.config = load_config()
        self._pipeline: RAGPipeline = None
        self._register_handlers()
    
    @property
    def pipeline(self) -> RAGPipeline:
        """延迟初始化 Pipeline"""
        if self._pipeline is None:
            logger.info("初始化 RAG Pipeline...")
            self._pipeline = RAGPipeline(self.config)
        return self._pipeline
    
    def _register_handlers(self):
        """注册 MCP 工具处理器"""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """返回可用的工具列表"""
            return [
                Tool(
                    name="ingest_document",
                    description="上传文档到知识库",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_path": {"type": "string"},
                            "recursive": {"type": "boolean", "default": False}
                        },
                        "required": ["document_path"]
                    }
                ),
                Tool(
                    name="ask_question",
                    description="向知识库提问",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "session_id": {"type": "string"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="search_knowledge",
                    description="搜索知识库",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
            """执行工具调用"""
            logger.info(f"执行工具: {name}, 参数: {arguments}")
            
            try:
                if name == "ingest_document":
                    result = await self._handle_ingest(arguments)
                elif name == "ask_question":
                    result = await self._handle_ask(arguments)
                elif name == "search_knowledge":
                    result = await self._handle_search(arguments)
                else:
                    raise ValueError(f"未知工具: {name}")
                
                return [TextContent(type="text", text=result)]
            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                return [TextContent(type="text", text=f"错误: {str(e)}")]
    
    async def _handle_ingest(self, args: dict) -> str:
        """处理文档上传"""
        document_path = args["document_path"]
        await self.pipeline.build_index(document_path)
        return f"✅ 文档上传成功: {document_path}"
    
    async def _handle_ask(self, args: dict) -> str:
        """处理问答"""
        result = await self.pipeline.ask(
            args["question"],
            args.get("session_id")
        )
        
        output = f"💡 **回答**\n\n{result['answer']}\n\n"
        output += "**参考来源**:\n"
        for i, source in enumerate(result["sources"][:3], 1):
            preview = source["content"][:200] + "..."
            output += f"\n{i}. [{source['score']:.2f}] {preview}\n"
        
        return output
    
    async def _handle_search(self, args: dict) -> str:
        """处理检索"""
        results = await self.pipeline.search(
            args["query"],
            args.get("top_k", 10)
        )
        
        output = f"🔍 检索结果: \"{args['query']}\"\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. [{r['score']:.3f}] {r['content'][:300]}...\n\n"
        
        return output
    
    async def run_stdio(self):
        """STDIO 模式运行"""
        logger.info("启动 MCP Server (STDIO 模式)")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """主入口"""
    server = RAGMCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add src/mcp_server/server.py
git commit -m "feat: implement MCP server with three tools"
```

---

## Task 11: Evaluation with RAGAS

**Goal:** Add RAGAS evaluation framework

**Files:**
- Create: `src/evaluation/base.py`
- Create: `src/evaluation/ragas_evaluator.py`

- [ ] **Step 1: Create base.py**

```python
# src/evaluation/base.py
"""
评估器基类 - 评估 RAG 系统质量
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class EvalResult:
    """评估结果"""
    query: str
    answer: str
    contexts: List[str]
    metrics: Dict[str, float]


class BaseEvaluator(ABC):
    """评估器抽象基类"""
    
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: str = None
    ) -> EvalResult:
        """评估单个查询"""
        pass
```

- [ ] **Step 2: Create ragas_evaluator.py**

```python
# src/evaluation/ragas_evaluator.py
"""
RAGAS 评估器实现

【RAGAS 指标】
- Faithfulness: 回答是否忠实于上下文
- Answer Relevancy: 回答是否与问题相关
- Context Precision: 上下文精确率
- Context Recall: 上下文召回率
"""

from typing import List, Dict, Any

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

from src.evaluation.base import BaseEvaluator, EvalResult


class RagasEvaluator(BaseEvaluator):
    """RAGAS 评估器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    async def evaluate(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: str = None
    ) -> EvalResult:
        """使用 RAGAS 评估"""
        # 构建数据集
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # 执行评估
        scores = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        
        return EvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            metrics={k: v[0] for k, v in scores.items()}
        )
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/*.py
git commit -m "feat: add RAGAS evaluation framework"
```

---

## Task 12: Integration and Testing

**Goal:** Integration tests and final verification

**Files:**
- Create: `tests/integration/test_mcp_server.py`
- Create: `tests/conftest.py`
- Modify: `README.md`

- [ ] **Step 1: Create conftest.py**

```python
# tests/conftest.py
"""pytest 配置和共享 fixtures"""

import pytest


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "siliconflow_api_key": "test-key",
        "qianfan_api_key": "test-key",
        "qianfan_secret_key": "test-secret",
        "chunk_size": 512,
        "top_k": 10
    }
```

- [ ] **Step 2: Create integration test**

```python
# tests/integration/test_mcp_server.py
"""MCP Server 集成测试"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """测试 MCP 服务器初始化"""
    # 验证服务器能正确启动
    from src.mcp_server.server import RAGMCPServer
    
    server = RAGMCPServer()
    assert server is not None
    assert server.server is not None
```

- [ ] **Step 3: Update README.md**

```markdown
# RAG-MCP-Demo

基于 LlamaIndex 的 RAG 知识库系统，支持 MCP 协议

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export QIANFAN_API_KEY="your-key"
export QIANFAN_SECRET_KEY="your-secret"
export SILICONFLOW_API_KEY="your-key"
```

### 3. 运行 MCP Server

```bash
# STDIO 模式
python -m src.mcp_server.server
```

## 功能特性

- ✅ 文档上传（支持 txt, md, pdf, docx, xlsx）
- ✅ 混合检索（向量 + BM25）
- ✅ 智能问答（千帆 LLM）
- ✅ MCP 协议支持
- ✅ RAGAS 评估

## 架构

- **RAG 框架**: LlamaIndex
- **向量数据库**: Chroma (本地)
- **嵌入模型**: SiliconFlow
- **LLM**: 千帆 (ERNIE-Bot-4)
- **协议**: MCP (STDIO/SSE)
```

- [ ] **Step 4: Final commit**

```bash
git add tests/conftest.py tests/integration/*.py README.md
git commit -m "test: add integration tests and documentation"
```

---

## Summary

**Complete implementation plan with 12 tasks:**

| Task | Component | Files |
|------|-----------|-------|
| 1 | Project Bootstrap | requirements.txt, config/ |
| 2 | Utilities | config.py, logger.py, registry.py |
| 3 | LLM | base.py, qianfan_llm.py |
| 4 | Embedder | base.py, siliconflow_embedder.py |
| 5 | Vector Store | base.py, chroma_store.py |
| 6 | Loaders | base.py, office_loader.py |
| 7 | Reranker | base.py, siliconflow_reranker.py |
| 8 | Hybrid Retriever | hybrid_retriever.py |
| 9 | Main Pipeline | pipeline.py |
| 10 | MCP Server | server.py |
| 11 | Evaluation | base.py, ragas_evaluator.py |
| 12 | Integration | conftest.py, tests/, README.md |

**Total estimated time:** 6-8 hours

**Ready for execution using:**
- `superpowers:subagent-driven-development` (recommended)
- `superpowers:executing-plans` (inline)