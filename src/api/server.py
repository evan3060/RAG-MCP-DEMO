"""
RAG HTTP API 服务器 - 基于 FastAPI

提供 RESTful API 供前端调用：
- POST /api/ingest - 文档上传
- POST /api/ask - 智能问答
- POST /api/search - 知识库搜索
- GET /api/health - 健康检查

使用方法:
    uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============ 延迟导入 RAG 组件（避免在模块加载时初始化） ============
_mcp_server: Optional = None

def get_mcp_server():
    """延迟初始化 MCP Server"""
    global _mcp_server
    if _mcp_server is None:
        # 启用嵌套事件循环（必须在 LlamaIndex 导入前应用）
        import nest_asyncio
        nest_asyncio.apply()

        from src.mcp_server.server import RAGMCPServer
        _mcp_server = RAGMCPServer()
    return _mcp_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 初始化 RAG MCP Server...")
    server = get_mcp_server()
    print("✅ 服务器初始化完成")
    yield
    print("🛑 服务器关闭")


app = FastAPI(
    title="RAG MCP API",
    description="RAG 知识库系统 HTTP API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 请求/响应模型 ============

class IngestRequest(BaseModel):
    """文档上传请求"""
    document_path: str
    recursive: bool = False


class IngestResponse(BaseModel):
    """文档上传响应"""
    success: bool
    message: str
    document_path: str


class AskRequest(BaseModel):
    """问答请求"""
    question: str
    session_id: Optional[str] = None


class Source(BaseModel):
    """参考来源"""
    content: str
    score: float
    metadata: dict


class AskResponse(BaseModel):
    """问答响应"""
    success: bool
    answer: str
    sources: list[Source]
    session_id: Optional[str]


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    """搜索结果"""
    content: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    query: str
    results: list[SearchResult]
    total: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    llm_provider: str
    embedding_model: str


# ============ API 端点 ============

@app.get("/", response_model=dict)
async def root():
    """根路径 - API 信息"""
    return {
        "name": "RAG MCP API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "/api/ingest",
            "ask": "/api/ask",
            "search": "/api/search",
            "health": "/api/health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    server = get_mcp_server()
    config = server.config
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=config.get("llm", {}).get("provider", "unknown"),
        embedding_model=config.get("embedding", {}).get("siliconflow", {}).get("model", "unknown")
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    上传文档到知识库

    示例请求:
    ```json
    {
        "document_path": "./knowledge_base",
        "recursive": true
    }
    ```
    """
    try:
        server = get_mcp_server()
        result_text = await server._handle_ingest({
            "document_path": request.document_path,
            "recursive": request.recursive
        })

        return IngestResponse(
            success=True,
            message=result_text,
            document_path=request.document_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    向知识库提问

    示例请求:
    ```json
    {
        "question": "什么是人工智能？",
        "session_id": "session_001"
    }
    ```
    """
    try:
        server = get_mcp_server()
        # 调用 pipeline 获取结构化数据
        pipeline_result = await server.pipeline.ask(
            request.question,
            request.session_id
        )

        return AskResponse(
            success=True,
            answer=pipeline_result["answer"],
            sources=[
                Source(
                    content=s["content"],
                    score=s["score"],
                    metadata=s["metadata"]
                )
                for s in pipeline_result["sources"]
            ],
            session_id=pipeline_result.get("session_id")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    搜索知识库

    示例请求:
    ```json
    {
        "query": "人工智能应用",
        "top_k": 5
    }
    ```
    """
    try:
        server = get_mcp_server()
        results = await server.pipeline.search(
            request.query,
            request.top_k
        )

        return SearchResponse(
            success=True,
            query=request.query,
            results=[
                SearchResult(
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in results
            ],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
