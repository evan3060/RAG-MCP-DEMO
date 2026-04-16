"""MCP Server 集成测试"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """测试 MCP 服务器初始化"""
    from src.mcp_server.server import RAGMCPServer

    server = RAGMCPServer()
    assert server is not None
    assert server.server is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tools_list():
    """测试 MCP 工具列表"""
    from src.mcp_server.server import RAGMCPServer

    server = RAGMCPServer()
    tools = await server.server.list_tools()

    tool_names = [tool.name for tool in tools]
    assert "ingest_document" in tool_names
    assert "ask_question" in tool_names
    assert "search_knowledge" in tool_names
