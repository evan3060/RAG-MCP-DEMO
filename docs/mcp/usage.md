# MCP 协议使用指南

本文档详细介绍 MCP（Model Context Protocol）协议以及如何在 RAG-MCP-DEMO 中使用它。

## 什么是 MCP？

MCP（Model Context Protocol，模型上下文协议）是一种标准化协议，用于 AI 模型与外部工具和服务进行通信。它类似于 API，但专门为 AI 场景设计，让 AI 能够调用各种工具来完成任务。

### MCP 的核心概念

| 概念 | 说明 |
|------|------|
| **Server** | 提供工具的服务端（本项目） |
| **Client** | 调用工具的客户端（如 OpenCode、Claude Desktop） |
| **Tools** | 服务端提供的可调用工具 |
| **Transport** | 传输层（STDIO 或 SSE） |

## RAG-MCP-DEMO 提供的工具

本项目通过 MCP 协议提供以下三个工具：

### 1. ingest_document - 文档摄入

将文档加载到知识库中。

**参数**：
| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `document_path` | string | 是 | 文档所在目录路径 |
| `recursive` | boolean | 否 | 是否递归扫描子目录（默认 false） |

**返回值**：成功消息或错误信息

**示例调用**：
```json
{
  "name": "ingest_document",
  "arguments": {
    "document_path": "./knowledge_base",
    "recursive": true
  }
}
```

### 2. ask_question - 智能问答

基于知识库回答用户问题。

**参数**：
| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `question` | string | 是 | 用户问题 |
| `session_id` | string | 否 | 会话ID，用于保持上下文 |

**返回值**：包含答案和参考来源的 JSON

**示例调用**：
```json
{
  "name": "ask_question",
  "arguments": {
    "question": "什么是 RAG？",
    "session_id": "user_123_session_456"
  }
}
```

**返回格式**：
```json
{
  "answer": "RAG 是检索增强生成（Retrieval Augmented Generation）的缩写...",
  "sources": [
    {
      "content": "RAG是一种结合检索和生成的技术...",
      "score": 0.95,
      "metadata": {"file_name": "rag_intro.txt"}
    }
  ],
  "session_id": "user_123_session_456"
}
```

### 3. search_knowledge - 知识检索

纯语义检索，不调用 LLM 生成答案。

**参数**：
| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `query` | string | 是 | 查询关键词 |
| `top_k` | integer | 否 | 返回结果数量（默认 10） |

**返回值**：检索结果列表

**示例调用**：
```json
{
  "name": "search_knowledge",
  "arguments": {
    "query": "RAG 技术原理",
    "top_k": 5
  }
}
```

## 传输模式

RAG-MCP-DEMO 支持两种传输模式：

### 1. STDIO 模式（标准输入输出）

适合本地调用和命令行工具集成。

**启动命令**：
```bash
python -m src.mcp_server.server
```

**工作原理**：
```
stdin  ←─── MCP 协议消息（JSON）
stdout ──── MCP 协议响应（JSON）
```

**使用场景**：
- 命令行直接调用
- 与 OpenCode 集成
- 作为子进程调用

### 2. SSE 模式（Server-Sent Events）

适合 Web 应用和远程调用。

**启动命令**：
```bash
# 需要设置环境变量
export MCP_TRANSPORT=sse
export MCP_PORT=8080
python -m src.mcp_server.server
```

**访问地址**：`http://localhost:8080/mcp`

**使用场景**：
- Web 应用集成
- 远程 API 调用
- 需要通过 HTTP 访问的场景

## 与 OpenCode 集成

OpenCode 支持 MCP 协议，可以通过配置将 RAG-MCP-DEMO 作为工具集成。

### 步骤一：创建 MCP 配置文件

在项目根目录创建 `.opencode/mcp.json`：

```json
{
  "mcp": {
    "rag-server": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### 步骤二：在 OpenCode 中使用

配置完成后，OpenCode 会自动发现并注册 MCP 工具。你可以直接说：

```
"请上传 knowledge_base 目录中的文档到知识库"
```

或

```
"根据知识库回答：RAG 技术有哪些应用场景？"
```

## 与 Claude Desktop 集成

### 步骤一：安装 Claude Desktop

从 [Claude Desktop 官网](https://claude.com/desktop) 下载安装。

### 步骤二：配置 MCP 服务器

编辑 Claude Desktop 配置文件：

- **macOS**：`~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**：`~/.config/Claude/claude_desktop_config.json`
- **Windows**：`%APPDATA%\Claude\claude_desktop_config.json`

添加配置：

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "PYTHONPATH": "/path/to/RAG-MCP-DEMO"
      }
    }
  }
}
```

### 步骤三：重启 Claude Desktop

重启后，你可以在对话中直接使用 MCP 工具。

## Python 代码调用示例

### 基本调用

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 配置服务器参数
    params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server.server"],
        env={"PYTHONPATH": "."}
    )
    
    # 创建客户端连接
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()
            
            # 调用工具：摄入文档
            result = await session.call_tool("ingest_document", {
                "document_path": "./knowledge_base"
            })
            print(result)
            
            # 调用工具：问答
            result = await session.call_tool("ask_question", {
                "question": "什么是 RAG？"
            })
            print(result)

asyncio.run(main())
```

### 完整示例

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class RAGMCPClient:
    """RAG MCP 客户端"""
    
    def __init__(self, python_path: str = "python"):
        self.python_path = python_path
        self.session = None
    
    async def __aenter__(self):
        params = StdioServerParameters(
            command=self.python_path,
            args=["-m", "src.mcp_server.server"],
            env={"PYTHONPATH": "."}
        )
        
        self.read, self.write = await stdio_client(params).__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.initialize()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.__aexit__(*args)
    
    async def ingest_document(self, path: str, recursive: bool = False):
        """摄入文档"""
        result = await self.session.call_tool("ingest_document", {
            "document_path": path,
            "recursive": recursive
        })
        return result
    
    async def ask_question(self, question: str, session_id: str = None):
        """智能问答"""
        params = {"question": question}
        if session_id:
            params["session_id"] = session_id
        result = await self.session.call_tool("ask_question", params)
        return result
    
    async def search(self, query: str, top_k: int = 10):
        """知识检索"""
        result = await self.session.call_tool("search_knowledge", {
            "query": query,
            "top_k": top_k
        })
        return result


# 使用示例
async def main():
    async with RAGMCPClient() as client:
        # 摄入文档
        print(await client.ingest_document("./knowledge_base"))
        
        # 问答
        result = await client.ask_question("RAG 技术有什么优势？")
        print(result)
        
        # 检索
        result = await client.search("向量检索", top_k=5)
        print(result)

asyncio.run(main())
```

## HTTP API 调用（SSE 模式）

如果使用 SSE 模式，可以通过 HTTP 调用：

```python
import requests
import sseclient
import json

def ask_question(question: str, session_id: str = None):
    """通过 HTTP API 提问"""
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ask_question",
            "arguments": {
                "question": question,
                "session_id": session_id
            }
        }
    }
    
    response = requests.post(
        "http://localhost:8080/mcp",
        json=payload,
        stream=True
    )
    
    # 处理 SSE 响应
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.data:
            print(event.data)

# 使用
ask_question("什么是 RAG？")
```

## 错误处理

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| `-32600` | 无效请求 | 检查 JSON 格式是否正确 |
| `-32601` | 方法不存在 | 检查工具名称是否正确 |
| `-32602` | 参数无效 | 检查参数类型和值 |
| `-32000` | 服务器错误 | 查看错误消息，检查日志 |

### 错误响应格式

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": "详细错误信息"
  },
  "id": null
}
```

## 性能优化建议

### 1. 批量摄入文档

不要每次调用 ingest_document，而是批量处理：

```python
# 不推荐：多次调用
for doc in documents:
    await client.ingest_document(doc)

# 推荐：一次性摄入整个目录
await client.ingest_document("./knowledge_base", recursive=True)
```

### 2. 复用会话

保持会话连接以减少建立连接的开销：

```python
# 复用同一个客户端会话
async with RAGMCPClient() as client:
    # 多个问答复用同一会话
    for question in questions:
        result = await client.ask_question(question)
```

### 3. 调整检索参数

根据实际需求调整 top_k 参数：

```python
# 只需要少量参考来源
result = await client.search("query", top_k=3)

# 需要更多参考来源
result = await client.search("query", top_k=20)
```

## 下一步

- 想要学习如何扩展系统？查看 [扩展开发指南](../development/)
- 想要查看 API 参考？查看 [API 参考文档](../api/)
- 遇到问题了？查看 [常见问题解答](../faq/)