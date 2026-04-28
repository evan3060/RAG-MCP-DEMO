# API 参考文档

本文档提供 RAG-MCP-DEMO 所有公共 API 的详细参考。

## 目录

1. [RAGPipeline API](#1-ragpipeline-api)
2. [SmartTextProcessor API](#2-smarttextprocessor-api)
3. [HybridRetriever API](#3-hybridretriever-api)
4. [MCP Server API](#4-mcp-server-api)
5. [配置 API](#5-配置-api)
6. [组件基类 API](#6-组件基类-api)

---

## 1. RAGPipeline API

**文件位置**：`src/rag/llamaindex/pipeline.py`

### 类：RAGPipeline

RAG 系统的主类，负责文档索引构建和问答处理。

#### 构造函数

```python
def __init__(self, config: Dict[str, Any])
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | Dict[str, Any] | 配置字典，通过 `load_config()` 获取 |

**示例**：
```python
from src.utils.config import load_config
from src.rag.llamaindex.pipeline import RAGPipeline

config = load_config()
pipeline = RAGPipeline(config)
```

#### 方法：build_index

```python
async def build_index(self, documents_path: str) -> VectorStoreIndex
```

构建知识库索引。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `documents_path` | str | 文档所在目录路径 |

**返回**：
- `VectorStoreIndex`：LlamaIndex 向量索引对象

**示例**：
```python
index = await pipeline.build_index("./knowledge_base")
print(f"索引包含 {len(index.docstore)} 个文档")
```

#### 方法：ask

```python
async def ask(
    self, 
    question: str, 
    session_id: Optional[str] = None,
    selected_files: Optional[List[str]] = None
) -> Dict[str, Any]
```

智能问答，基于知识库回答问题。

**参数**：
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `question` | str | 是 | 用户问题 |
| `session_id` | str | 否 | 会话ID，用于保持上下文 |
| `selected_files` | List[str] | 否 | 限定的知识库文件列表 |

**返回**：
```python
{
    "answer": "这是答案内容...",
    "sources": [
        {
            "content": "参考文档内容...",
            "score": 0.95,
            "metadata": {"file_name": "doc.txt"}
        }
    ],
    "session_id": "session_123"
}
```

**示例**：
```python
result = await pipeline.ask("什么是 RAG？")
print(result["answer"])
for source in result["sources"]:
    print(f"来源: {source['metadata']['file_name']} (相似度: {source['score']:.2f})")
```

#### 方法：search

```python
async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]
```

纯语义检索，不调用 LLM。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | str | 查询关键词 |
| `top_k` | int | 返回结果数量，默认 10 |

**返回**：
```python
[
    {
        "content": "检索到的内容...",
        "score": 0.92,
        "metadata": {"file_name": "doc.txt"}
    }
]
```

**示例**：
```python
results = await pipeline.search("RAG 技术", top_k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['content'][:100]}...")
```

---

## 2. SmartTextProcessor API

**文件位置**：`src/rag/llamaindex/pipeline.py` (第 28-251 行)

### 类：SmartTextProcessor

智能文本处理器，负责文档的预处理和分块。

#### 构造函数

```python
def __init__(self, doc_type: str = 'auto')
```

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `doc_type` | str | `'auto'` | 文档类型：`'general'`、`'technical'` 或 `'auto'` |

**分块参数**：
| 类型 | 最小字数 | 最大字数 |
|------|----------|----------|
| general | 200 | 500 |
| technical | 500 | 800 |

#### 方法：process

```python
def process(self, text: str, metadata: Dict = None) -> List[TextNode]
```

处理文本，返回智能切分后的节点列表。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | str | 待处理的原始文本 |
| `metadata` | Dict | 附加的元数据字典 |

**返回**：
- `List[TextNode]`：处理后的文本节点列表

**示例**：
```python
processor = SmartTextProcessor(doc_type='auto')
nodes = processor.process(
    "这是文档内容...",
    metadata={"file_name": "test.txt", "file_type": "text"}
)
print(f"处理后得到 {len(nodes)} 个节点")
```

---

## 3. HybridRetriever API

**文件位置**：`src/rag/llamaindex/hybrid_retriever.py`

### 类：HybridRetriever

混合检索器，结合向量检索和 BM25 关键词检索。

#### 构造函数

```python
def __init__(
    self,
    index: VectorStoreIndex,
    vector_retriever: VectorIndexRetriever,
    top_k: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
)
```

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `index` | VectorStoreIndex | - | LlamaIndex 索引 |
| `vector_retriever` | VectorIndexRetriever | - | 向量检索器 |
| `top_k` | int | 10 | 返回结果数量 |
| `vector_weight` | float | 0.7 | 向量检索权重 |
| `bm25_weight` | float | 0.3 | BM25 检索权重 |

#### 方法：aretrieve

```python
async def aretrieve(self, query: str) -> List[NodeWithScore]
```

异步检索方法。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | str | 查询字符串 |

**返回**：
- `List[NodeWithScore]`：带分数的检索结果列表

**示例**：
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from src.rag.llamaindex.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    index=index,
    vector_retriever=vector_retriever,
    top_k=10,
    vector_weight=0.7,
    bm25_weight=0.3
)

results = await retriever.aretrieve("RAG 技术")
for node in results:
    print(f"[{node.score:.3f}] {node.node.text[:100]}...")
```

### 函数：rrf_fusion

```python
def rrf_fusion(
    vector_results: List[NodeWithScore],
    bm25_results: List[NodeWithScore],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60
) -> List[NodeWithScore]
```

RRF（Reciprocal Rank Fusion）融合算法。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `vector_results` | List[NodeWithScore] | 向量检索结果 |
| `bm25_results` | List[NodeWithScore] | BM25 检索结果 |
| `vector_weight` | float | 向量检索权重 |
| `bm25_weight` | float | BM25 检索权重 |
| `k` | int | RRF 平滑参数，默认 60 |

**返回**：
- `List[NodeWithScore]`：融合后的结果列表

---

## 4. MCP Server API

**文件位置**：`src/mcp_server/server.py`

### 类：RAGMCPServer

MCP 协议服务器实现。

#### 构造函数

```python
def __init__(self)
```

**示例**：
```python
from src.mcp_server.server import RAGMCPServer

server = RAGMCPServer()
```

#### 属性：pipeline

```python
@property
def pipeline(self) -> RAGPipeline
```

获取 RAG Pipeline 实例（懒加载）。

**返回**：
- `RAGPipeline`：RAG 管道实例

#### 方法：run_stdio

```python
async def run_stdio(self)
```

以 STDIO 模式运行 MCP 服务器。

**示例**：
```python
import asyncio
from src.mcp_server.server import RAGMCPServer

async def main():
    server = RAGMCPServer()
    await server.run_stdio()

asyncio.run(main())
```

### MCP 工具定义

#### 1. ingest_document

```python
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
)
```

#### 2. ask_question

```python
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
)
```

#### 3. search_knowledge

```python
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
```

---

## 5. 配置 API

**文件位置**：`src/utils/config.py`

### 函数：load_config

```python
def load_config() -> Dict[str, Any]
```

加载项目配置。

**返回**：
```python
{
    "llm": {
        "provider": "qianfan",
        "qianfan": {...},
        "siliconflow": {...},
        "openai": {...}
    },
    "embedding": {
        "provider": "siliconflow",
        "siliconflow": {...}
    },
    "reranker": {
        "provider": "siliconflow",
        "siliconflow": {...}
    },
    "vector_store": {
        "provider": "chroma",
        "persist_dir": "./data/chroma_db"
    }
}
```

**示例**：
```python
from src.utils.config import load_config

config = load_config()
print(f"LLM 提供商: {config['llm']['provider']}")
```

---

## 6. 组件基类 API

### BaseLLM

**文件位置**：`src/rag/components/llms/base.py`

```python
class BaseLLM(ABC):
    """LLM 抽象基类"""
    
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """同步聊天"""
        pass
    
    @abstractmethod
    async def achat(self, messages: list, **kwargs) -> str:
        """异步聊天"""
        pass
    
    @abstractmethod
    def stream_chat(self, messages: list, **kwargs) -> Iterator[str]:
        """流式聊天"""
        pass
```

### BaseEmbedder

**文件位置**：`src/rag/components/embedders/base.py`

```python
class BaseEmbedder(ABC):
    """嵌入模型抽象基类"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量表示"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取向量表示"""
        pass
```

### BaseVectorStore

**文件位置**：`src/rag/components/vector_stores/base.py`

```python
class BaseVectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add(self, nodes: List[TextNode]):
        """添加节点"""
        pass
    
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[TextNode]:
        """查询"""
        pass
    
    @abstractmethod
    def delete(self, node_ids: List[str]):
        """删除节点"""
        pass
```

### BaseLoader

**文件位置**：`src/rag/components/loaders/base.py`

```python
class BaseLoader(ABC):
    """文档加载器抽象基类"""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文档扩展名"""
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """加载文档"""
        pass
```

---

## 错误码参考

| 错误码 | 类 | 说明 |
|--------|------|------|
| 1001 | ValueError | 索引未构建 |
| 1002 | FileNotFoundError | 文档路径不存在 |
| 1003 | ImportError | 依赖未安装 |
| 1004 | KeyError | 配置项缺失 |
| 1005 | ConnectionError | API 连接失败 |

---

## 下一步

- 遇到问题了？查看 [常见问题解答](../faq/)
- 想要学习完整流程？查看 [核心模块分析](../deep-dive/)
- 想要扩展系统？查看 [扩展开发指南](../development/)