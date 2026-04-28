# 核心模块源码分析

本文档对 RAG-MCP-DEMO 的核心模块进行深入的源码分析，帮助开发者理解系统的工作原理。

## 目录

1. [RAGPipeline 流程](#1-ragpipeline-流程)
2. [SmartTextProcessor 智能文本处理](#2-smarttextprocessor-智能文本处理)
3. [HybridRetriever 混合检索](#3-hybridretriever-混合检索)
4. [MCP Server 实现](#4-mcp-server-实现)
5. [配置加载机制](#5-配置加载机制)

---

## 1. RAGPipeline 流程

**文件位置**：`src/rag/llamaindex/pipeline.py`

RAGPipeline 是整个 RAG 系统的核心类，负责文档索引构建和问答处理。

### 1.1 类结构

```python
class RAGPipeline:
    """RAG Pipeline 主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index: Optional[VectorStoreIndex] = None
        self._configure_settings()    # 配置 LlamaIndex 全局设置
        self._load_existing_index()   # 尝试加载已有索引
```

**初始化流程**：
1. 接收配置字典
2. 配置 LlamaIndex 全局设置（LLM、嵌入模型）
3. 尝试加载已存在的 Chroma 索引

### 1.2 构建索引

```python
async def build_index(self, documents_path: str) -> VectorStoreIndex:
    """构建知识库索引"""
    
    # 1. 配置文档加载器
    file_extractor = {
        ".pdf": PDFLoader(),
        ".docx": DocxLoader(),
        ".xlsx": ExcelLoader(),
    }
    
    # 2. 加载文档
    documents = SimpleDirectoryReader(
        documents_path,
        file_extractor=file_extractor
    ).load_data()
    
    # 3. 智能文本处理
    processor = SmartTextProcessor(doc_type='auto')
    all_nodes = []
    for doc in documents:
        nodes = processor.process(doc.text, metadata={...})
        all_nodes.extend(nodes)
    
    # 4. 存储到 Chroma
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection("knowledge_base")
    vector_store = LlamaIndexChromaStore(chroma_collection=collection)
    
    # 5. 创建索引
    self.index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_context)
    return self.index
```

### 1.3 问答流程

```python
async def ask(self, question: str, session_id: Optional[str] = None):
    """智能问答"""
    
    # 1. 创建检索器
    vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=20)
    hybrid_retriever = HybridRetriever(index=self.index, vector_retriever=vector_retriever, top_k=10)
    
    # 2. 创建聊天引擎
    chat_engine = ContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000)
    )
    
    # 3. 获取回复
    response = await chat_engine.achat(question)
    
    # 4. 过滤思考过程
    answer = self._filter_thinking_process(response.response)
    
    # 5. 获取参考来源
    vector_results = await vector_retriever.aretrieve(question)
    sources = [...]
    
    return {"answer": answer, "sources": sources, "session_id": session_id}
```

### 1.4 关键设计点

| 设计点 | 说明 |
|--------|------|
| **异步方法** | 使用 `async/await` 提高并发性能 |
| **索引持久化** | Chroma 支持磁盘持久化，避免重复构建 |
| **会话记忆** | 使用 `ChatMemoryBuffer` 保留对话上下文 |
| **思考过滤** | 移除 LLM 的思考过程，只返回最终答案 |

---

## 2. SmartTextProcessor 智能文本处理

**文件位置**：`src/rag/llamaindex/pipeline.py` (第 28-251 行)

SmartTextProcessor 负责文档的预处理和智能分块。

### 2.1 功能概述

```python
class SmartTextProcessor:
    """智能文本处理器"""
    
    CHUNK_SIZES = {
        'general': (200, 500),      # 通用文档：200-500 字
        'technical': (500, 800),    # 技术文档：500-800 字
    }
    OVERLAP_RATIO = 0.15  # 15% 重叠率
```

### 2.2 处理流程

```python
def process(self, text: str, metadata: Dict = None) -> List[TextNode]:
    """处理文本，返回智能切分后的节点列表"""
    
    # 步骤1：基础清洗
    text = self._basic_clean(text)
    
    # 步骤2：检测文档类型
    if self.doc_type == 'auto':
        self._detect_doc_type(text)
    
    # 步骤3：结构解析
    blocks = self._parse_structure(text)
    
    # 步骤4：智能分块
    nodes = self._smart_chunk(blocks, metadata)
    
    return nodes
```

### 2.3 文本清洗

```python
def _basic_clean(self, text: str) -> str:
    """基础文本清洗"""
    
    # 1. 移除控制字符
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    
    # 2. 换行处理：单换行→空格，双换行→分段
    text = text.replace('\n\n', '\x00PARA\x00')
    text = text.replace('\n', ' ')
    text = text.replace('\x00PARA\x00', '\n\n')
    
    # 3. 清理页眉页脚、页码
    text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
    text = re.sub(r'(第\s*\d+\s*页|Page\s*\d+\s*of\s*\d+)', '', text, flags=re.IGNORECASE)
    
    return text.strip()
```

### 2.4 文档类型检测

```python
def _detect_doc_type(self, text: str):
    """自动检测文档类型"""
    
    TECHNICAL_KEYWORDS = [
        '代码', '函数', 'api', '接口', '算法', '实现', '配置',
        '代码', 'function', 'def ', 'import ', 'class'
    ]
    
    # 统计技术关键词出现次数
    tech_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw.lower() in text.lower())
    
    # 根据关键词密度判断文档类型
    if tech_score > len(text) / 5000:
        self.doc_type = 'technical'
        self.chunk_size = self.CHUNK_SIZES['technical']
    else:
        self.doc_type = 'general'
        self.chunk_size = self.CHUNK_SIZES['general']
```

### 2.5 智能分块算法

```python
def _smart_chunk(self, blocks: List[Dict], metadata: Dict = None) -> List[TextNode]:
    """智能分块，保持结构完整性"""
    
    nodes = []
    current_chunk = []
    current_size = 0
    min_size, max_size = self.chunk_size
    
    for block in blocks:
        block_len = len(block['content'])
        
        # 结构性块（标题、代码、表格）单独处理
        if block['is_structural']:
            if current_chunk:
                nodes.append(self._create_node(current_chunk, metadata))
                current_chunk = []
            nodes.append(self._create_node([block], metadata, is_structural=True))
            continue
        
        # 普通段落：检查是否超出最大大小
        if current_size + block_len > max_size and current_size >= min_size:
            # 保存当前块
            nodes.append(self._create_node(current_chunk, metadata))
            
            # 应用重叠：保留上一块的结尾部分
            overlap_blocks = self._calculate_overlap(current_chunk)
            current_chunk = overlap_blocks + [block]
            current_size = sum(len(b['content']) for b in current_chunk)
        else:
            current_chunk.append(block)
            current_size += block_len
    
    # 处理最后一个块
    if current_chunk:
        nodes.append(self._create_node(current_chunk, metadata))
    
    return nodes
```

### 2.6 分块策略说明

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **段落完整** | 不拆分段落，保持语义完整 | 普通文档 |
| **标题绑定** | 标题与后续内容放在同一块 | 结构性文档 |
| **结构性块独立** | 标题、代码块、表格单独成块 | 技术文档 |
| **重叠机制** | 15% 重叠，保持上下文连贯 | 所有文档 |

---

## 3. HybridRetriever 混合检索

**文件位置**：`src/rag/llamaindex/hybrid_retriever.py`

HybridRetriever 结合向量检索和 BM25 关键词检索，提供更准确的检索结果。

### 3.1 检索流程

```python
class HybridRetriever:
    def __init__(self, index, vector_retriever, top_k=10, 
                 vector_weight=0.7, bm25_weight=0.3):
        self.index = index
        self.vector_retriever = vector_retriever
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    async def aretrieve(self, query: str) -> List[NodeWithScore]:
        # 1. 向量检索
        vector_results = await self.vector_retriever.aretrieve(query)
        
        # 2. BM25 检索（懒加载）
        bm25_retriever = self._get_bm25_retriever()
        if bm25_retriever is None:
            return vector_results[:self.top_k]
        
        bm25_results = await bm25_retriever.aretrieve(query)
        
        # 3. RRF 融合
        fused_results = rrf_fusion(
            vector_results, bm25_results,
            self.vector_weight, self.bm25_weight
        )
        
        return fused_results[:self.top_k]
```

### 3.2 RRF 融合算法

```python
def rrf_fusion(vector_results, bm25_results, 
               vector_weight=0.7, bm25_weight=0.3, k=60):
    """RRF (Reciprocal Rank Fusion) 融合算法"""
    
    scores = defaultdict(float)
    node_map = {}
    
    # 向量检索结果计分
    for rank, node in enumerate(vector_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += vector_weight / (k + rank)
        node_map[node_id] = node.node
    
    # BM25 检索结果计分
    for rank, node in enumerate(bm25_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += bm25_weight / (k + rank)
        if node_id not in node_map:
            node_map[node_id] = node.node
    
    # 按得分排序
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [NodeWithScore(node=node_map[nid], score=score) 
            for nid, score in sorted_nodes]
```

### 3.3 权重配置

| 权重配置 | 向量检索 | BM25 | 说明 |
|----------|----------|------|------|
| 默认 | 0.7 | 0.3 | 向量检索权重更高 |
| 平衡 | 0.5 | 0.5 | 两种检索平等对待 |
| 关键词优先 | 0.3 | 0.7 | 适用于精确匹配场景 |

### 3.4 为什么混合检索效果更好

| 检索方式 | 优点 | 缺点 |
|----------|------|------|
| **向量检索** | 语义相似、容忍拼写错误 | 可能遗漏精确关键词 |
| **BM25** | 精确匹配、速度快 | 无法处理同义词和语义 |

RRF 融合结合两者的优点，既能处理语义相似性，又能保证关键词匹配。

---

## 4. MCP Server 实现

**文件位置**：`src/mcp_server/server.py`

MCP Server 实现了 Model Context Protocol 协议，提供工具调用接口。

### 4.1 服务器类结构

```python
class RAGMCPServer:
    """RAG MCP 服务器"""
    
    def __init__(self):
        self.server = Server("rag-mcp-server")
        self.config = load_config()
        self._pipeline: RAGPipeline = None
        self._register_handlers()
```

### 4.2 工具注册

```python
def _register_handlers(self):
    """注册 MCP 工具处理器"""
    
    @self.server.list_tools()
    async def list_tools() -> list[Tool]:
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
                inputSchema={...}
            ),
            Tool(
                name="search_knowledge",
                description="搜索知识库",
                inputSchema={...}
            )
        ]
```

### 4.3 工具调用处理

```python
@self.server.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
    """处理工具调用请求"""
    
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
```

### 4.4 STDIO 模式启动

```python
async def run_stdio(self):
    """以 STDIO 模式运行服务器"""
    
    logger.info("启动 MCP Server (STDIO 模式)")
    async with stdio_server() as (read_stream, write_stream):
        await self.server.run(
            read_stream,
            write_stream,
            self.server.create_initialization_options()
        )

async def main():
    server = RAGMCPServer()
    await server.run_stdio()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.5 MCP 工具说明

| 工具名称 | 功能 | 参数 |
|----------|------|------|
| `ingest_document` | 上传文档到知识库 | `document_path`: 文档路径 |
| `ask_question` | 智能问答 | `question`: 问题, `session_id`: 会话ID |
| `search_knowledge` | 纯语义检索 | `query`: 查询词, `top_k`: 返回数量 |

---

## 5. 配置加载机制

**文件位置**：`src/utils/config.py`

### 5.1 配置加载流程

```python
def load_config() -> Dict[str, Any]:
    """加载配置"""
    
    # 1. 加载 .env 环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 2. 合并配置
    config = {
        "llm": _load_llm_config(),
        "embedding": _load_embedding_config(),
        "reranker": _load_reranker_config(),
        "vector_store": _load_vector_store_config(),
    }
    
    return config
```

### 5.2 配置优先级

```
环境变量 (.env) > 默认配置 > 代码硬编码
```

### 5.3 配置结构示例

```python
config = {
    "llm": {
        "provider": "qianfan",
        "qianfan": {
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL", "ERNIE-Bot-4"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "temperature": 0.7
        },
        "siliconflow": {...},
        "openai": {...}
    },
    "embedding": {
        "provider": "siliconflow",
        "siliconflow": {
            "api_key": os.getenv("EMBEDDING_API_KEY"),
            "model": "BAAI/bge-large-zh-v1.5"
        }
    },
    ...
}
```

---

## 核心类关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RAGMCPServer                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  - list_tools()           → 返回可用工具列表                    │ │
│  │  - call_tool()            → 分发请求到处理方法                  │ │
│  │  - _handle_ingest()       → 处理文档上传                        │ │
│  │  - _handle_ask()          → 处理问答请求                        │ │
│  │  - _handle_search()       → 处理检索请求                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│                    ┌───────────────────┐                            │
│                    │   RAGPipeline     │                            │
│                    │   - build_index() │                            │
│                    │   - ask()         │                            │
│                    │   - search()      │                            │
│                    └─────────┬─────────┘                            │
│                              │                                       │
│        ┌─────────────────────┼─────────────────────┐                │
│        ▼                     ▼                     ▼                │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐            │
│  │SmartText   │      │HybridRe-   │      │LLMAdapter  │            │
│  │Processor   │      │triever     │      │            │            │
│  │- 清洗      │      │- 向量检索  │      │- 千帆      │            │
│  │- 分块      │      │- BM25检索  │      │- SiliconFlow│           │
│  │- 类型检测  │      │- RRF融合   │      │- OpenAI    │            │
│  └──────┬─────┘      └──────┬─────┘      └────────────┘            │
│         │                  │                                       │
│         ▼                  ▼                                       │
│  ┌────────────┐      ┌────────────┐                                 │
│  │  TextNode  │      │VectorStore │                                 │
│  │  (片段)    │      │ (Chroma)   │                                 │
│  └────────────┘      └────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 扩展点说明

| 扩展点 | 位置 | 说明 |
|--------|------|------|
| **新增 LLM** | `src/rag/components/llms/` | 实现 `BaseLLM` 接口 |
| **新增 Embedder** | `src/rag/components/embedders/` | 实现 `BaseEmbedding` 接口 |
| **新增 Loader** | `src/rag/components/loaders/` | 实现文档加载逻辑 |
| **新增 Vector Store** | `src/rag/components/vector_stores/` | 实现向量存储 |
| **新增 MCP 工具** | `src/mcp_server/server.py` | 在 `list_tools()` 中注册 |

---

## 下一步

- 想要学习如何扩展系统？查看 [扩展开发指南](../development/)
- 想要了解 MCP 协议？查看 [MCP 协议使用指南](../mcp/)
- 想要查看 API 参考？查看 [API 参考文档](../api/)