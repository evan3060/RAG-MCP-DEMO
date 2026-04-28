# 架构设计文档

本文档详细介绍 RAG-MCP-DEMO 的系统架构，帮助你理解各个组件之间的关系以及数据流动方式。

## 系统概述

RAG-MCP-DEMO 是一个基于 LlamaIndex 的检索增强生成（RAG）系统，支持 MCP（Model Context Protocol）协议。系统的主要功能是将文档加载到向量数据库中，然后通过语义搜索和 LLM 生成答案。

### 核心特性

- **文档加载**：支持多种格式（txt, md, pdf, docx, xlsx）
- **混合检索**：结合向量检索和 BM25 关键词检索
- **智能问答**：基于检索结果生成答案
- **MCP 协议**：支持 STDIO 和 SSE 两种传输模式

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户 / MCP 客户端                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (src/mcp_server/)                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  RAGMCPServer                                                │ │
│  │  - ingest_document (上传文档)                                  │ │
│  │  - ask_question (智能问答)                                     │ │
│  │  - search_knowledge (纯检索)                                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RAG Pipeline (src/rag/llamaindex/)               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ SmartTextProcessor│  │ HybridRetriever │  │  RAGPipeline    │  │
│  │ - 文本清洗        │  │ - 向量检索       │  │ - 索引构建       │  │
│  │ - 智能分块        │  │ - BM25检索       │  │ - 问答处理       │  │
│  │ - 结构解析        │  │ - RRF融合        │  │ - 答案过滤       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Document       │  │  Vector Store   │  │  LLM            │
│  Loaders        │  │  (Chroma)       │  │  (千帆/SF/OpenAI)│
│  - PDF          │  │                 │  │                 │
│  - Docx         │  │  存储向量数据    │  │  生成答案        │
│  - Excel        │  │  持久化到磁盘    │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│  Embedder       │                │  Reranker       │
│  (BGE嵌入模型)   │                │  (BGE重排序)    │
│  文本→向量       │                │  结果重排序      │
└─────────────────┘                └─────────────────┘
```

## 核心模块详解

### 1. MCP Server 模块

**位置**：`src/mcp_server/server.py`

MCP Server 是整个系统的入口，负责接收来自 MCP 客户端的请求并调用相应的处理逻辑。

#### 主要功能

| 功能 | 方法 | 说明 |
|------|------|------|
| 列出工具 | `list_tools()` | 返回可用工具列表 |
| 调用工具 | `call_tool()` | 分发请求到具体处理方法 |

#### 提供的工具

1. **ingest_document** - 文档摄入
   - 参数：`document_path`（必需）、`recursive`（可选）
   - 功能：将指定路径的文档加载到向量数据库

2. **ask_question** - 智能问答
   - 参数：`question`（必需）、`session_id`（可选）
   - 功能：基于知识库回答用户问题

3. **search_knowledge** - 知识检索
   - 参数：`query`（必需）、`top_k`（可选，默认 10）
   - 功能：纯语义检索，不调用 LLM

### 2. RAG Pipeline 模块

**位置**：`src/rag/llamaindex/pipeline.py`

RAG Pipeline 是核心业务逻辑层，负责处理文档加载、索引构建、问答查询等操作。

#### 类：`RAGPipeline`

**主要方法**：

| 方法 | 说明 |
|------|------|
| `__init__` | 初始化配置和加载已有索引 |
| `build_index()` | 构建知识库索引 |
| `ask()` | 智能问答 |
| `search()` | 纯检索 |

#### 类：`SmartTextProcessor`

智能文本处理器，负责文档的预处理和分块。

**功能**：
- 文本清洗：移除控制字符、页眉页脚、页码水印
- 文档类型检测：自动识别技术文档或普通文档
- 结构解析：识别标题、列表、表格、代码块
- 智能分块：根据内容类型选择合适的块大小

**分块策略**：
- 通用文档：200-500 字
- 技术文档：500-800 字
- 重叠率：15%（保持上下文连贯）

### 3. 混合检索模块

**位置**：`src/rag/llamaindex/hybrid_retriever.py`

混合检索器结合了向量检索和 BM25 关键词检索的优点。

#### 检索流程

```
用户查询
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  向量检索        │     │  BM25 检索       │
│  (Vector Store) │     │  (关键词匹配)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │   RRF 融合算法       │
         │ (Reciprocal Rank    │
         │    Fusion)          │
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │   返回 Top K 结果   │
         └─────────────────────┘
```

#### RRF 融合算法

RRF（Reciprocal Rank Fusion）是一种简单而有效的多检索结果融合算法：

```python
score(doc_id) = Σ (weight_i / (k + rank_i))
```

其中：
- `weight_i` : 第 i 个检索器的权重（向量检索 0.7，BM25 0.3）
- `rank_i` : 文档在第 i 个检索结果中的排名
- `k` : 常数（默认 60），用于平滑排名差异

### 4. 组件层

**位置**：`src/rag/components/`

组件层提供了可替换的模块化组件，支持多种提供商。

#### LLM 组件

| 组件 | 文件 | 支持提供商 |
|------|------|------------|
| 千帆 LLM | `llms/qianfan_llm.py` | 百度千帆 |
| SiliconFlow LLM | `llms/siliconflow_llm.py` | SiliconFlow |
| OpenAI 兼容 LLM | `llms/openai_compatible_llm.py` | OpenAI、兼容 API |

#### Embedder 组件

| 组件 | 文件 | 支持提供商 |
|------|------|------------|
| SiliconFlow Embedder | `embedders/siliconflow_embedder.py` | SiliconFlow |
| LlamaIndex Adapter | `embedders/llama_index_adapter.py` | 适配各种嵌入模型 |

#### Reranker 组件

| 组件 | 文件 | 支持模型 |
|------|------|----------|
| SiliconFlow Reranker | `rerankers/siliconflow_reranker.py` | BAAI/bge-reranker-v2-m3 |

#### Vector Store 组件

| 组件 | 文件 | 说明 |
|------|------|------|
| Chroma Store | `vector_stores/chroma_store.py` | Chroma 向量数据库 |

#### Document Loader 组件

| 组件 | 文件 | 支持格式 |
|------|------|----------|
| PDF Loader | `loaders/pdf_loader.py` | PDF |
| Office Loader | `loaders/office_loader.py` | Word (docx), Excel (xlsx/xls) |

### 5. 工具模块

**位置**：`src/utils/`

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置管理 | `config.py` | 加载和管理配置 |
| 日志管理 | `logger.py` | 统一的日志输出 |
| 注册中心 | `registry.py` | 组件注册和发现 |

## 数据流

### 文档摄入流程

```
1. 用户调用 ingest_document
         │
         ▼
2. MCP Server 接收请求
         │
         ▼
3. RAGPipeline.build_index()
         │
         ▼
4. Document Loader 加载文档
   (PDFLoader / DocxLoader / ExcelLoader)
         │
         ▼
5. SmartTextProcessor 处理文本
   (清洗、分块、结构解析)
         │
         ▼
6. Embedder 生成向量
   (BAAI/bge-large-zh-v1.5)
         │
         ▼
7. Chroma Vector Store 存储
         │
         ▼
8. 返回成功消息
```

### 问答流程

```
1. 用户提问
         │
         ▼
2. MCP Server 接收问题
         │
         ▼
3. RAGPipeline.ask()
         │
         ▼
4. HybridRetriever 检索
   - 向量检索 (Top 20)
   - BM25 检索 (Top 20)
   - RRF 融合 (Top 10)
         │
         ▼
5. LLM 生成答案
   (千帆 / SiliconFlow / OpenAI)
         │
         ▼
6. 过滤思考过程
         │
         ▼
7. 返回答案和参考来源
```

## 配置结构

项目配置通过 `src/utils/config.py` 加载管理：

```python
config = {
    "llm": {
        "provider": "qianfan",  # 或 "siliconflow", "openai"
        "qianfan": {
            "api_key": "...",
            "model": "ERNIE-Bot-4",
            "base_url": "..."
        }
    },
    "embedding": {
        "provider": "siliconflow",
        "siliconflow": {
            "api_key": "...",
            "model": "BAAI/bge-large-zh-v1.5"
        }
    },
    "reranker": {
        "provider": "siliconflow",
        "siliconflow": {
            "api_key": "...",
            "model": "BAAI/bge-reranker-v2-m3"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "persist_dir": "./data/chroma_db"
    }
}
```

## 扩展开发

系统采用模块化设计，方便扩展新功能：

1. **新增 LLM 提供商**：在 `src/rag/components/llms/` 下实现新类
2. **新增文档格式**：在 `src/rag/components/loaders/` 下实现 Loader
3. **新增向量数据库**：在 `src/rag/components/vector_stores/` 下实现
4. **新增 MCP 工具**：在 `src/mcp_server/server.py` 中注册

详见 [扩展开发指南](../development/)。

## 技术选型理由

| 组件 | 选型 | 理由 |
|------|------|------|
| RAG 框架 | LlamaIndex | 功能丰富、社区活跃、文档完善 |
| 向量数据库 | Chroma | 轻量级、易部署、支持持久化 |
| 嵌入模型 | BGE | 开源免费、中文效果优秀 |
| 重排序模型 | BGE Reranker | 显著提升检索效果 |
| LLM | 千帆/SiliconFlow | 国内可用、性价比高 |
| 协议 | MCP | 标准化的 AI 工具交互协议 |

## 性能优化

1. **索引持久化**：Chroma 支持持久化，避免重复构建索引
2. **懒加载 BM25**：BM25 检索器在首次检索时才初始化
3. **结果缓存**：会话内存保留最近对话上下文
4. **思考过程过滤**：移除 LLM 的思考过程，减少输出噪音

## 下一步

- 想要查看某个模块的源码详解？查看 [核心模块分析](../deep-dive/)
- 想要学习如何扩展系统？查看 [扩展开发指南](../development/)
- 想要了解 MCP 协议？查看 [MCP 协议使用指南](../mcp/)