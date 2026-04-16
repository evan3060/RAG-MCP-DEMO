# RAG-MCP-Demo 设计文档

**日期**: 2026-04-16  
**作者**: Claude  
**状态**: 待审核  

---

## 1. 项目概述

### 1.1 目标

构建一个面向初级开发者的 RAG (Retrieval-Augmented Generation) 学习项目，实现以下核心功能：

- 用户上传文档构建知识库
- 基于混合检索的智能问答
- 纯检索模式（不调用 LLM）
- 以 MCP (Model Context Protocol) 服务形式提供
- 支持 STDIO 和 SSE 两种通信模式

### 1.2 目标用户

**初级开发人员**：需要通过详细的代码注释、示例和文档来学习 RAG 架构、LlamaIndex 框架和 MCP 协议。

### 1.3 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| 后端框架 | Python + FastAPI | 异步支持，生态丰富 |
| RAG 框架 | LlamaIndex | 专为 RAG 设计，学习价值高 |
| LLM | 千帆 (qianfan) | 百度智能云，国内访问稳定 |
| 向量数据库 | SiliconFlow | 提供 Embedding 和向量存储 |
| 重排序 | SiliconFlow (免费) | BAAI/bge-reranker-v2-m3 |
| MCP 协议 | stdio + sse | 双模式支持 |
| 评估框架 | RAGAS + LlamaIndex | 自动化评估 |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MCP Server (API 网关层)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ ingest_document │  │  ask_question   │  │search_knowledge │     │
│  │   (文档上传)     │  │   (智能问答)     │  │   (知识检索)     │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
└───────────┼────────────────────┼────────────────────┼──────────────┘
            │                    │                    │
            └────────────────────┴──────────┬─────────┘
                                              │
┌─────────────────────────────────────────────▼───────────────────────┐
│                           RAG Pipeline Core                          │
│                                                                      │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│   │ IngestionPipeline │    │RetrievalPipeline │    │EvalPipeline  │ │
│   │ • LlamaIndex      │    │ • Hybrid Search  │    │ • RAGAS     │ │
│   │ • Semantic Split │    │ • Reranker       │    │ • DeepEval  │ │
│   └──────────────────┘    └──────────────────┘    └──────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
rag-mcp-demo/
├── src/
│   ├── mcp_server/           # MCP 服务层
│   │   ├── server.py         # 主入口
│   │   ├── stdio.py          # STDIO 模式
│   │   └── sse.py            # SSE 模式
│   ├── rag/
│   │   ├── llamaindex/       # LlamaIndex 实现
│   │   │   ├── pipeline.py   # 主 Pipeline
│   │   │   └── hybrid_retriever.py
│   │   └── components/       # 可插拔组件
│   │       ├── loaders/
│   │       ├── embedders/
│   │       ├── rerankers/
│   │       └── llms/
│   ├── evaluation/           # 评估模块
│   └── docs/                 # 学习文档
├── tests/
├── examples/                 # 示例代码
└── config/
```

---

## 3. MCP 工具设计

### 3.1 工具列表

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `ingest_document` | 上传文档构建索引 | document_path, recursive | 处理结果 |
| `ask_question` | 智能问答 | question, session_id? | 回答 + 来源 |
| `search_knowledge` | 纯检索 | query, top_k? | 文档片段列表 |

### 3.2 工具对比

| 特性 | `ask_question` | `search_knowledge` |
|------|----------------|-------------------|
| LLM 调用 | ✅ 有 | ❌ 无 |
| 会话支持 | ✅ 有 | ❌ 无 |
| 返回内容 | 生成答案 | 原始片段 |
| 适用场景 | 直接获取答案 | 查看原始文档 |

---

## 4. RAG Pipeline 设计

### 4.1 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           文档上传流程 (Ingestion)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Local File                                                          │
│        ↓                                                              │
│   LlamaIndex SimpleDirectoryReader                                    │
│        ↓                                                              │
│   TextNode (语义切分)                                                  │
│        ↓                                                              │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  SiliconFlow Embedding API (云端)                            │   │
│   │  文本 → 向量 [0.023, -0.156, 0.892, ...] 1024维             │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ↓                                        │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  Chroma Vector Store (本地)                                  │   │
│   │  • 向量: [0.023, -0.156, ...]                                │   │
│   │  • 原文: "RAG 是一种检索增强..."                              │   │
│   │  • 元数据: {source: "doc.md", chunk_index: 3}                │   │
│   │  • 存储: ./data/chroma_db/ (本地文件，自动持久化)              │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   💡 可切换至 Qdrant/Milvus (修改配置即可)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           查询流程 (Query)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Query: "什么是RAG?"                                             │
│        ↓                                                              │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  SiliconFlow Embedding API                                   │   │
│   │  "什么是RAG?" → 查询向量                                      │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ↓                                        │
│   ┌──────────────────┐    ┌──────────────────┐                       │
│   │ Dense Retriever  │    │ BM25 Retriever   │                       │
│   │ (向量相似度)      │    │ (关键词匹配)      │                       │
│   │ Chroma: cosine   │    │ 本地倒排索引      │                       │
│   │ Top-20 候选      │    │ Top-20 候选      │                       │
│   └────────┬─────────┘    └────────┬─────────┘                       │
│            ↓                      ↓                                   │
│            └──────────┬───────────┘                                   │
│                       ↓                                               │
│              RRF Fusion (混合排序)                                     │
│                       ↓                                               │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  SiliconFlow Rerank API (云端重排序)                         │   │
│   │  Top-10 → Top-5 (更精准)                                     │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ↓                                        │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  千帆 LLM (qianfan)                                          │   │
│   │  Context + Question → Answer                                 │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ↓                                        │
│                    Generated Answer                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 文档加载器插件设计

#### 4.2.1 支持的文档格式

| 格式 | 扩展名 | 依赖库 | 优先级 | 说明 |
|------|--------|--------|--------|------|
| **文本** | .txt, .md | 内置 | P0 | 基础支持 |
| **PDF** | .pdf | pymupdf | P0 | 已有支持 |
| **Word** | .docx | python-docx | P0 | Office 主流格式 |
| **Excel** | .xlsx, .xls | openpyxl, xlrd | P0 | 表格数据 |
| **PPT** | .pptx | python-pptx | P1 | 演示文稿 |
| **旧 Word** | .doc | antiword/pywin32 | P2 | 需额外工具 |

#### 4.2.2 Office 加载器设计

```python
# src/rag/components/loaders/office_loader.py
"""Office 文档加载器 - 支持 Word、Excel、PowerPoint"""

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class DocxLoader(BaseReader):
    """Word 文档加载器 (.docx)
    
    解析内容:
    - 段落文本
    - 表格内容（转换为 Markdown 格式）
    - 文档属性（作者、标题、创建时间）
    """
    
    def load_data(self, file_path: Path) -> List[Document]:
        from docx import Document as DocxDocument
        
        doc = DocxDocument(file_path)
        
        # 提取段落
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # 提取表格（转为 Markdown 格式，LLM 更易理解）
        tables_md = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.replace('|', '\\|') for cell in row.cells]
                rows.append('| ' + ' | '.join(cells) + ' |')
            # 添加表头分隔线
            if rows:
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
            'author': doc.core_properties.author,
        }
        
        return [Document(text=content, metadata=metadata)]


class ExcelLoader(BaseReader):
    """Excel 加载器 (.xlsx, .xls)
    
    特殊处理:
    - 多 Sheet 分别处理
    - 转换为 Markdown 表格（LLM 理解更好）
    - 限制超大表格（默认 10000 行）
    """
    
    def __init__(self, max_rows: int = 10000):
        self.max_rows = max_rows
    
    def load_data(self, file_path: Path) -> List[Document]:
        import pandas as pd
        
        xl_file = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in xl_file.sheet_names:
            df = xl_file.parse(sheet_name, nrows=self.max_rows)
            
            # 转换为 Markdown 表格
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


class UnifiedOfficeLoader(BaseReader):
    """统一 Office 加载器 - 自动识别文件类型"""
    
    LOADERS = {
        '.docx': DocxLoader,
        '.xlsx': ExcelLoader,
        '.xls': ExcelLoader,
        # '.pptx': PptxLoader,  # 第二阶段实现
    }
    
    def load_data(self, file_path: Path) -> List[Document]:
        suffix = Path(file_path).suffix.lower()
        loader_class = self.LOADERS.get(suffix)
        
        if not loader_class:
            raise ValueError(f'不支持的格式: {suffix}')
        
        return loader_class().load_data(file_path)
```

#### 4.2.3 配置更新

```yaml
# config/default.yaml - 文档处理配置更新
ingestion:
  # 扩展支持的文件格式
  supported_extensions: [
    ".txt", ".md",                    # 文本
    ".pdf",                           # PDF
    ".docx",                          # Word (P0)
    ".xlsx", ".xls",                  # Excel (P0)
    # ".pptx",                        # PowerPoint (P1，后续支持)
    # ".doc",                         # 旧 Word (P2，需额外工具)
  ]
  
  office:
    excel:
      max_rows_per_sheet: 10000       # 防止超大表格内存溢出
      date_format: "%Y-%m-%d"
    
    word:
      extract_tables: true            # 提取表格为 Markdown
      include_headers: false          # 不提取页眉
```

#### 4.2.4 安装依赖

```bash
# requirements-office.txt
# Office 文档支持（可选安装）
python-docx>=0.8.11      # Word 支持
openpyxl>=3.0.0          # Excel .xlsx
xlrd>=2.0.0              # Excel .xls
# python-pptx>=0.6.21    # PPT 支持 (第二阶段)
```

**安装命令**:
```bash
# 基础安装（不含 Office）
pip install -r requirements.txt

# 完整安装（含 Office 支持）
pip install -r requirements.txt -r requirements-office.txt
```

### 4.3 向量数据库插件设计

**默认实现**: Chroma (本地嵌入式)

```python
# src/rag/components/vector_stores/chroma_store.py
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class ChromaVectorStore(BaseVectorStore):
    """Chroma 本地向量数据库实现
    
    特点:
    - 零部署: pip install chromadb 即可使用
    - 自动持久化: 数据保存到本地文件
    - 适合: 开发、Demo、小团队 (<10万文档)
    """
    
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
```

**可切换实现**:

| 数据库 | 适用场景 | 切换方式 |
|--------|---------|---------|
| Chroma | 开发/Demo (默认) | `provider: chroma` |
| Qdrant | 高性能本地/远程 | `provider: qdrant` |
| Milvus | 企业级生产 | `provider: milvus` |
| Weaviate | 多模态支持 | `provider: weaviate` |

**切换示例** (仅修改配置，零代码改动):

```yaml
# 从 Chroma 切换到 Qdrant
vector_store:
  provider: "qdrant"
  qdrant:
    path: "./data/qdrant_storage"  # 本地模式
    collection_name: "knowledge_base"
```

### 4.3 核心组件

#### 4.3.1 文档加载 (Loader)
- **SimpleDirectoryReader**: LlamaIndex 内置（文本、PDF）
- **OfficeLoader**: 自定义实现（Word、Excel，详见 4.2）
- **支持格式**: .txt, .md, .pdf, .docx, .xlsx, .xls
- **扩展点**: 实现 BaseLoader 接口可添加新格式

#### 4.3.2 文本切分 (Splitter)
- **SemanticSplitterNodeParser**: 语义切分（推荐）
- **SentenceSplitter**: 简单切分（备选）
- **配置**: chunk_size=512, chunk_overlap=50

#### 4.3.3 嵌入模型 (Embedder)
- **SiliconFlowEmbedding**: BAAI/bge-large-zh-v1.5
- **备选**: BAAI/bge-m3 (多语言), bge-small (轻量)

#### 4.3.4 混合检索 (Hybrid Retriever)
- **Vector Retriever**: 语义相似，Dense Embedding
- **BM25 Retriever**: 关键词匹配，Sparse
- **融合算法**: RRF (Reciprocal Rank Fusion)
- **权重**: vector=0.7, bm25=0.3

#### 4.3.5 重排序 (Reranker)
- **模型**: BAAI/bge-reranker-v2-m3 (SiliconFlow 免费)
- **作用**: 精排 Top-K，提升准确性
- **输出**: 最终送入 LLM 的片段

#### 4.3.6 LLM 生成
- **千帆**: ERNIE-Bot-4
- **温度**: 0.7
- **模式**: ContextChatEngine (支持会话)

---

## 5. 插件系统设计

### 5.1 设计目标

实现组件级别的可插拔，支持：
- LLM 可切换（千帆/OpenAI/Azure/Ollama）
- Embedding 可切换（SiliconFlow/OpenAI/本地）
- Retriever 可切换（Hybrid/Dense/Sparse）
- 未来扩展方便

### 5.2 抽象基类

每个组件定义抽象基类：
- `BaseLLM`: generate(), stream_generate()
- `BaseEmbedder`: embed(), embed_batch()
- `BaseVectorStore`: upsert(), search(), delete() - **向量数据库基类（新增）**
- `BaseRetriever`: retrieve(), aretrieve()
- `BaseReranker`: rerank()

**向量数据库插件设计**:
```python
# src/rag/components/vector_stores/base.py
class BaseVectorStore(ABC):
    """向量数据库抽象基类 - 支持多种后端"""
    
    @abstractmethod
    async def upsert(self, ids, embeddings, documents, metadatas): ...
    
    @abstractmethod
    async def search(self, query_embedding, top_k, filters): ...
    
    @abstractmethod
    async def delete(self, ids): ...

# 具体实现
class ChromaVectorStore(BaseVectorStore): ...      # 本地默认
class QdrantVectorStore(BaseVectorStore): ...      # 可选
class MilvusVectorStore(BaseVectorStore): ...      # 可选
```

### 5.3 注册机制

```python
# 配置驱动实例化
llm = LLMRegistry.create(config["llm"]["provider"], config)
```

---

## 6. 评估体系

### 6.1 评估框架

**RAGAS** (主要):
- Faithfulness (忠实度)
- Answer Relevancy (答案相关性)
- Context Precision (上下文精确率)
- Context Recall (上下文召回率)

**LlamaIndex Eval** (辅助):
- 与框架深度集成
- 快速验证

### 6.2 评估流程

```
测试用例 → RAG Pipeline → 收集结果 → RAGAS 评估 → 生成报告
```

---

## 7. 配置设计

### 7.1 配置文件

`config/default.yaml` - 完整配置示例：

```yaml
# ==================== 服务配置 ====================
server:
  # MCP 服务运行模式: stdio | sse
  mode: stdio
  
  # SSE 模式配置
  host: "0.0.0.0"
  port: 8000

# ==================== LLM 配置 ====================
llm:
  # 可切换: qianfan | openai | azure_openai | ollama
  provider: "qianfan"
  
  qianfan:
    api_key: "${QIANFAN_API_KEY}"
    secret_key: "${QIANFAN_SECRET_KEY}"
    model: "ERNIE-Bot-4"
    temperature: 0.7

# ==================== 嵌入模型配置 ====================
embedding:
  # 使用 SiliconFlow 云端嵌入
  provider: "siliconflow"
  
  siliconflow:
    api_key: "${SILICONFLOW_API_KEY}"
    model: "BAAI/bge-large-zh-v1.5"

# ==================== 向量数据库配置 ====================
vector_store:
  # 可切换: chroma (默认本地) | qdrant | milvus | weaviate
  provider: "chroma"
  
  # Chroma 本地配置（默认推荐）
  chroma:
    persist_directory: "./data/chroma_db"
    collection_name: "knowledge_base"
    distance_fn: "cosine"  # cosine | l2 | ip
  
  # Qdrant 配置（可选，高性能）
  # qdrant:
  #   path: "./data/qdrant_storage"  # 本地模式
  #   # url: "http://localhost:6333"  # 远程模式
  #   collection_name: "knowledge_base"
  #   vector_size: 1024

# ==================== 检索配置 ====================
retrieval:
  strategy: "hybrid"  # hybrid | dense | sparse
  top_k: 10
  
  hybrid:
    vector_weight: 0.7
    bm25_weight: 0.3
    fusion_mode: "rrf"  # rrf | linear

# ==================== 重排序配置 ====================
reranker:
  enabled: true
  siliconflow:
    api_key: "${SILICONFLOW_API_KEY}"
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 5

# ==================== 文档处理配置 ====================
ingestion:
  supported_extensions: [".txt", ".md", ".pdf"]
  
  splitter:
    strategy: "semantic"  # semantic | recursive
    semantic:
      buffer_size: 1
      breakpoint_percentile_threshold: 95

# ==================== 评估配置 ====================
evaluation:
  enabled: true
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
```

### 7.2 插件配置切换示例

**切换到 Qdrant（高性能）**:
```yaml
vector_store:
  provider: "qdrant"
  qdrant:
    path: "./data/qdrant_storage"
    collection_name: "knowledge_base"
```

**切换到 OpenAI（备选）**:
```yaml
llm:
  provider: "openai"
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
```

### 7.3 环境变量

```bash
# 必需
export QIANFAN_API_KEY="your-key"
export QIANFAN_SECRET_KEY="your-secret"
export SILICONFLOW_API_KEY="your-key"

# 可选（备选方案）
export OPENAI_API_KEY="your-key"
export QDRANT_API_KEY="your-key"  # 远程 Qdrant 使用
```

---

## 8. Web API 接口（前端集成预留）

> **说明**: 当前阶段专注后端核心功能实现。Web 前端可后续基于以下 API 开发。

SSE 模式下，系统提供 RESTful API 供前端调用：

### 8.1 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/ingest` | 上传文档 |
| POST | `/api/v1/ask` | 智能问答 |
| POST | `/api/v1/search` | 知识检索 |
| GET | `/api/v1/health` | 健康检查 |

### 8.2 请求/响应示例

**上传文档**:
```bash
POST /api/v1/ingest
Content-Type: multipart/form-data

file: document.pdf
metadata: {"category": "技术文档"}
```

**智能问答**:
```bash
POST /api/v1/ask
Content-Type: application/json

{
  "question": "什么是RAG?",
  "session_id": "optional-session-id"
}

Response:
{
  "answer": "RAG是检索增强生成...",
  "sources": [...],
  "session_id": "xxx"
}
```

### 8.3 前端集成建议

后续前端开发可选方案：
- **React/Vue**: 功能完整的 Web 应用
- **Streamlit**: 快速原型验证
- **移动端**: React Native / Flutter

---

## 9. 错误处理

### 8.1 异常类型

| 异常 | 场景 | 处理 |
|------|------|------|
| `ConfigError` | 配置缺失或无效 | 启动时检查，给出明确提示 |
| `IndexNotFoundError` | 查询时索引未构建 | 提示用户先上传文档 |
| `LLMError` | LLM API 调用失败 | 重试 + 降级策略 |
| `RetrievalError` | 检索失败 | 返回空结果 + 日志 |

### 9.2 日志

- 级别: INFO (默认), DEBUG (调试)
- 格式: 时间 - 名称 - 级别 - 消息
- 输出: 控制台 + 文件

---

## 10. 学习资源设计

### 10.1 代码注释

所有核心代码包含详细注释，包括：
- 概念解释（类比、举例）
- 关键代码解释
- 参数说明
- 使用示例
- 设计模式说明

### 10.2 文档

`src/docs/`:
- 01_concepts.md: RAG 核心概念
- 02_llamaindex_guide.md: LlamaIndex 入门
- 03_mcp_protocol.md: MCP 协议详解
- 04_plugin_system.md: 插件系统设计
- 05_evaluation.md: 评估方法
- 06_troubleshooting.md: 常见问题

### 10.3 示例代码

`examples/`:
- 01_basic_ingestion.py: 基础文档上传
- 02_basic_query.py: 基础查询
- 03_chat_session.py: 对话会话
- 04_hybrid_search.py: 混合检索
- 05_custom_component.py: 自定义组件
- 06_evaluation.py: 评估示例

---

## 10. 部署

### 10.1 STDIO 模式

```bash
python -m src.mcp_server.server
# 或
python -m src.mcp_server.server --stdio
```

适用于：Claude Desktop、本地命令行工具

### 10.2 SSE 模式

```bash
python -m src.mcp_server.server --sse 0.0.0.0 8000
```

适用于：Web 应用、远程服务

### 10.3 Docker (可选)

提供 Dockerfile 支持容器化部署。

---

## 11. 测试策略

### 11.1 单元测试

- 组件级测试（Loader, Splitter, Retriever）
- Mock 外部 API
- 快速执行

### 11.2 集成测试

- Pipeline 端到端测试
- 需要真实 API 调用（标记为 slow）

### 11.3 评估测试

- 使用测试数据集验证质量
- 回归测试

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| API 限流/故障 | 高 | 实现重试 + 降级 |
| 大文档处理慢 | 中 | 异步处理 + 进度反馈 |
| 新手学习曲线 | 中 | 详细注释 + 文档 |
| 索引膨胀 | 低 | 定期清理/分区策略 |

---

## 13. 后续扩展

### 13.1 第一阶段 (当前)
- 基础 RAG + MCP
- 文本文档支持
- 基本评估

### 13.2 第二阶段
- 多文档类型 (PDF, Word, PPT)
- 用户权限隔离
- 高级评估指标

### 13.3 第三阶段
- 多模态 (图片、表格)
- 知识库版本管理
- 生产级部署

---

## 附录 A: 术语表

| 术语 | 解释 |
|------|------|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| MCP | Model Context Protocol，模型上下文协议 |
| Embedding | 嵌入，文本转向量的过程 |
| BM25 | 基于词频的检索算法 |
| Hybrid Search | 混合检索，结合多种检索方式 |
| Rerank | 重排序，对初步结果精排 |
| Chunk | 文档切分后的片段 |
| Node | LlamaIndex 中的文档单元 |
| Index | 索引，加速检索的数据结构 |

---

## 附录 B: 参考资源

- [LlamaIndex 文档](https://docs.llamaindex.ai)
- [RAGAS 文档](https://docs.ragas.io)
- [MCP 协议规范](https://modelcontextprotocol.io)
- [SiliconFlow API](https://siliconflow.cn)
- [千帆 API](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html)
