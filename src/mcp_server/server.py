"""MCP Server 实现 - Model Context Protocol 服务端"""

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
        if self._pipeline is None:
            logger.info("初始化 RAG Pipeline...")
            self._pipeline = RAGPipeline(self.config)
        return self._pipeline

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
        document_path = args["document_path"]
        await self.pipeline.build_index(document_path)
        return f"✅ 文档上传成功: {document_path}"

    async def _handle_ask(self, args: dict) -> str:
        result = await self.pipeline.ask(args["question"], args.get("session_id"))

        output = f"💡 **回答**\n\n{result['answer']}\n\n"
        output += "**参考来源**:\n"
        for i, source in enumerate(result["sources"][:3], 1):
            preview = source["content"][:200] + "..."
            output += f"\n{i}. [{source['score']:.2f}] {preview}\n"

        return output

    async def _handle_search(self, args: dict) -> str:
        results = await self.pipeline.search(args["query"], args.get("top_k", 10))

        output = f"🔍 检索结果: \"{args['query']}\"\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. [{r['score']:.3f}] {r['content'][:300]}...\n\n"

        return output

    async def run_stdio(self):
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
