# 扩展开发指南

本文档详细介绍如何为 RAG-MCP-DEMO 添加新功能，包括新的 LLM 提供商、文档格式支持、自定义检索器等。

## 目录

1. [架构概述](#1-架构概述)
2. [添加新的 LLM 提供商](#2-添加新的-llm-提供商)
3. [添加新的嵌入模型](#3-添加新的嵌入模型)
4. [添加新的文档加载器](#4-添加新的文档加载器)
5. [添加新的向量数据库](#5-添加新的向量数据库)
6. [添加新的 MCP 工具](#6-添加新的-mcp-工具)
7. [添加新的检索器](#7-添加新的检索器)

---

## 1. 架构概述

项目采用模块化设计，核心原则是**依赖倒置**和**接口抽象**：

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Server                            │
│                    (src/mcp_server/)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│                  (src/rag/llamaindex/)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  LLM Layer    │  │ Embedder Layer│  │  Storage      │
│  (接口抽象)   │  │ (接口抽象)    │  │  (接口抽象)   │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
   ┌────┴────┐         ┌────┴────┐        ┌────┴────┐
   ▼         ▼         ▼         ▼        ▼         ▼
实现1    实现2      实现1    实现2     实现1    实现2
```

---

## 2. 添加新的 LLM 提供商

### 2.1 实现步骤

假设我们要添加 **Anthropic Claude** 支持：

#### 步骤 1：创建 LLM 类

在 `src/rag/components/llms/` 目录下创建 `claude_llm.py`：

```python
"""Anthropic Claude LLM 实现"""

from typing import Optional, Dict, Any, Iterator
import anthropic

from src.rag.components.llms.base import BaseLLM


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-opus-20240229")
        self.base_url = config.get("base_url")  # 可选，用于代理
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        
        # 初始化客户端
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(self, messages: list, **kwargs) -> str:
        """同步聊天"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=messages
        )
        return response.content[0].text
    
    async def achat(self, messages: list, **kwargs) -> str:
        """异步聊天"""
        # Anthropic SDK 原生支持异步
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=messages
        )
        return response.content[0].text
    
    def stream_chat(self, messages: list, **kwargs) -> Iterator[str]:
        """流式聊天"""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
```

#### 步骤 2：创建适配器

创建 `src/rag/components/llms/claude_adapter.py`，使其适配 LlamaIndex 接口：

```python
"""Claude LLM LlamaIndex 适配器"""

from typing import Any, Optional
from llama_index.core.llms import LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from src.rag.components.llms.claude_llm import ClaudeLLM


class ClaudeAdapter(LLM):
    """Claude LLM 适配器"""
    
    def __init__(self, llm: ClaudeLLM):
        self.llm = llm
    
    @property
    def metadata(self) -> LLM.Metadata:
        return LLM.Metadata(
            model_name=self.llm.model,
            context_window=200000,
            num_output=self.llm.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )
    
    def chat(self, messages: list, **kwargs) -> LLM.ChatResponse:
        text = self.llm.chat(messages, **kwargs)
        return LLM.ChatResponse(
            message=LLM.ChatMessage(role="assistant", content=text),
            raw=text,
        )
    
    async def achat(self, messages: list, **kwargs) -> LLM.ChatResponse:
        text = await self.llm.achat(messages, **kwargs)
        return LLM.ChatResponse(
            message=LLM.ChatMessage(role="assistant", content=text),
            raw=text,
        )
    
    def complete(self, prompt: str, **kwargs) -> LLM.CompletionResponse:
        text = self.llm.chat([{"role": "user", "content": prompt}], **kwargs)
        return LLM.CompletionResponse(
            text=text,
            raw=text,
        )
    
    async def acomplete(self, prompt: str, **kwargs) -> LLM.CompletionResponse:
        text = await self.llm.achat([{"role": "user", "content": prompt}], **kwargs)
        return LLM.CompletionResponse(
            text=text,
            raw=text,
        )
```

#### 步骤 3：在 Pipeline 中注册

修改 `src/rag/llamaindex/pipeline.py`：

```python
# 在文件顶部导入
from src.rag.components.llms.claude_llm import ClaudeLLM
from src.rag.components.llms.claude_adapter import ClaudeAdapter

# 在 _configure_settings 方法中添加
elif llm_provider == "claude":
    claude_config = llm_config.get("claude", {})
    llm = ClaudeLLM({
        "api_key": claude_config.get("api_key"),
        "model": claude_config.get("model", "claude-3-opus-20240229"),
        "max_tokens": claude_config.get("max_tokens", 4096),
        "temperature": claude_config.get("temperature", 0.7)
    })
    Settings.llm = ClaudeAdapter(llm)
```

#### 步骤 4：添加配置支持

修改 `src/utils/config.py`，添加 Claude 配置加载逻辑：

```python
def _load_llm_config() -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "qianfan")
    
    config = {"provider": provider}
    
    if provider == "claude":
        config["claude"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),  # 新增环境变量
            "model": os.getenv("LLM_MODEL", "claude-3-opus-20240229"),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        }
    # ... 其他 provider
    
    return config
```

#### 步骤 5：添加环境变量模板

在 `.env.example` 中添加：

```bash
# Anthropic Claude 配置（可选）
# ANTHROPIC_API_KEY=your-anthropic-key
```

### 2.2 完整代码结构

```
src/rag/components/llms/
├── __init__.py
├── base.py              # 抽象基类（已有）
├── qianfan_llm.py       # 千帆实现（已有）
├── siliconflow_llm.py   # SiliconFlow 实现（已有）
├── openai_compatible_llm.py  # OpenAI 兼容实现（已有）
└── claude_llm.py        # 新增：Claude 实现
    └── claude_adapter.py    # 新增：LlamaIndex 适配器
```

---

## 3. 添加新的嵌入模型

### 3.1 实现步骤

与添加 LLM 类似，假设要添加 **OpenAI Text Embedding**：

#### 步骤 1：创建 Embedder 类

在 `src/rag/components/embedders/` 目录下创建 `openai_embedder.py`：

```python
"""OpenAI Embedder 实现"""

from typing import List, Optional
import numpy as np
from llama_index.core.embeddings import BaseEmbedding

from src.rag.components.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 嵌入模型"""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-embedding-3-small")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.dimensions = config.get("dimensions", 1536)
        
        import httpx
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量表示"""
        response = self.client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": text,
                "dimensions": self.dimensions
            }
        )
        data = response.json()
        return data["data"][0]["embedding"]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取向量表示"""
        response = self.client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimensions
            }
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class OpenAIEmbedderAdapter(BaseEmbedding):
    """OpenAI Embedder LlamaIndex 适配器"""
    
    def __init__(self, embedder: OpenAIEmbedder):
        self.embedder = embedder
        super().__init__(callback_manager=None, verify=False)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.get_embeddings(texts)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.get_embeddings(texts)
```

#### 步骤 2：在 Pipeline 中注册

修改 `src/rag/llamaindex/pipeline.py` 的 `_configure_settings` 方法：

```python
elif embed_provider == "openai":
    oai_config = embedding_config.get("openai", {})
    Settings.embed_model = OpenAIEmbedderAdapter(
        OpenAIEmbedder({
            "api_key": oai_config.get("api_key"),
            "model": oai_config.get("model", "text-embedding-3-small"),
            "dimensions": oai_config.get("dimensions", 1536)
        })
    )
```

---

## 4. 添加新的文档加载器

### 4.1 实现步骤

假设要添加 Markdown 文档支持（LlamaIndex 已支持，这里仅作示例）：

#### 步骤 1：创建 Loader 类

```python
"""Markdown 文档加载器"""

from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document
from src.rag.components.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器"""
    
    def __init__(self):
        self.supported_extensions = [".md", ".markdown"]
    
    def load(self, file_path: str) -> List[Document]:
        """加载 Markdown 文件"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 解析 Markdown 元数据（如果使用 frontmatter）
        metadata = self._parse_frontmatter(content)
        content = self._strip_frontmatter(content)
        
        # 提取标题作为文件名的补充
        title = self._extract_title(content)
        metadata["file_name"] = path.name
        metadata["title"] = title
        
        return [Document(text=content, metadata=metadata)]
    
    def _parse_frontmatter(self, content: str) -> Dict[str, Any]:
        """解析 frontmatter 元数据"""
        if not content.startswith("---"):
            return {}
        
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        
        # 解析 YAML
        import yaml
        try:
            return yaml.safe_load(parts[1]) or {}
        except:
            return {}
    
    def _strip_frontmatter(self, content: str) -> str:
        """移除 frontmatter"""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content
    
    def _extract_title(self, content: str) -> str:
        """提取第一个标题作为文档标题"""
        import re
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1) if match else ""
```

#### 步骤 2：注册 Loader

在 `build_index` 方法中添加：

```python
file_extractor = {
    ".pdf": PDFLoader(),
    ".docx": DocxLoader(),
    ".xlsx": ExcelLoader(),
    ".md": MarkdownLoader(),  # 新增
}
```

---

## 5. 添加新的向量数据库

### 5.1 实现步骤

假设要添加 **Qdrant** 向量数据库支持：

#### 步骤 1：创建 Vector Store 类

```python
"""Qdrant 向量数据库实现"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStore

from src.rag.components.vector_stores.base import BaseVectorStore


class QdrantStore(BaseVectorStore):
    """Qdrant 向量数据库"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6333)
        self.collection_name = config.get("collection", "knowledge_base")
        
        self.client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保集合存在"""
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    
    def add(self, nodes: List[TextNode]):
        """添加节点到向量数据库"""
        points = [
            PointStruct(
                id=node.node_id,
                vector=node.embedding,
                payload={
                    "text": node.text,
                    "metadata": node.metadata
                }
            )
            for node in nodes
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def query(self, query_embedding: List[float], top_k: int = 10) -> List[TextNode]:
        """查询向量数据库"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            TextNode(
                text=result.payload["text"],
                metadata=result.payload.get("metadata", {}),
                embedding=query_embedding
            )
            for result in results
        ]
    
    def delete(self, node_ids: List[str]):
        """删除节点"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=node_ids
        )
```

#### 步骤 2：注册 Vector Store

在配置中添加 `qdrant` provider，然后在 Pipeline 中添加支持。

---

## 6. 添加新的 MCP 工具

### 6.1 实现步骤

假设要添加一个 **get_stats** 工具，用于获取知识库统计信息：

#### 步骤 1：注册新工具

修改 `src/mcp_server/server.py`：

```python
@self.server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... 现有工具 ...
        
        Tool(
            name="get_stats",
            description="获取知识库统计信息",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]
```

#### 步骤 2：添加处理方法

```python
@self.server.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
    # ... 现有分支 ...
    
    elif name == "get_stats":
        result = await self._handle_stats(arguments)
    
    # ...

async def _handle_stats(self, args: dict) -> str:
    """处理统计信息请求"""
    # 获取集合统计信息
    collection = self.pipeline.get_collection()
    count = collection.count()
    
    return f"""
📊 知识库统计

- 文档总数: {count}
- 索引路径: {self.config.get("vector_store", {}).get("persist_dir", "./data/chroma_db")}
- 嵌入模型: {self.config.get("embedding", {}).get("siliconflow", {}).get("model", "BAAI/bge-large-zh-v1.5")}
"""
```

---

## 7. 添加新的检索器

### 7.1 实现步骤

假设要添加一个 **基于知识的检索器**，利用文档的元数据过滤：

#### 步骤 1：创建检索器类

```python
"""基于元数据过滤的检索器"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class MetadataFilterRetriever:
    """支持元数据过滤的检索器"""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        filters: dict,
        similarity_top_k: int = 10
    ):
        self.index = index
        self.filters = filters
        self.similarity_top_k = similarity_top_k
        self._base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k * 2  # 获取更多结果用于过滤
        )
    
    async def aretrieve(self, query: str) -> List[NodeWithScore]:
        # 1. 获取候选结果
        results = await self._base_retriever.aretrieve(query)
        
        # 2. 根据元数据过滤
        filtered = []
        for node in results:
            match = True
            for key, value in self.filters.items():
                if node.node.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(node)
            
            if len(filtered) >= self.similarity_top_k:
                break
        
        return filtered
```

#### 步骤 2：在 Pipeline 中使用

```python
# 在 ask 方法中
filters = {"file_type": "pdf"}  # 只检索 PDF 文档
retriever = MetadataFilterRetriever(
    index=self.index,
    filters=filters,
    similarity_top_k=10
)
```

---

## 测试和调试

### 单元测试

```python
# tests/test_new_llm.py

import pytest
from src.rag.components.llms.claude_llm import ClaudeLLM

def test_claude_llm_init():
    """测试 Claude LLM 初始化"""
    config = {
        "api_key": "test-key",
        "model": "claude-3-haiku-20240307"
    }
    llm = ClaudeLLM(config)
    assert llm.model == "claude-3-haiku-20240307"

@pytest.mark.asyncio
async def test_claude_llm_chat():
    """测试 Claude LLM 聊天"""
    # 需要 mock 或使用真实 API key
    pass
```

### 调试技巧

1. **查看日志**：设置 `LOG_LEVEL=DEBUG` 查看详细日志
2. **断点调试**：在 VS Code 中设置断点
3. **小数据集测试**：先用少量文档测试

---

## 下一步

- 想要查看 API 参考？查看 [API 参考文档](../api/)
- 遇到问题了？查看 [常见问题解答](../faq/)
- 想要学习完整流程？查看 [核心模块分析](../deep-dive/)