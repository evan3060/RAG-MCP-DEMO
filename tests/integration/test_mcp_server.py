"""MCP Server 集成测试"""

import pytest
from unittest.mock import patch


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """测试 MCP 服务器初始化"""
    with patch('src.mcp_server.server.load_config') as mock_config:
        mock_config.return_value = {
            "llm": {"provider": "qianfan"},
            "embedding": {"provider": "siliconflow"},
            "reranker": {"provider": "siliconflow"},
            "vector_store": {"provider": "chroma"}
        }

        from src.mcp_server.server import RAGMCPServer

        server = RAGMCPServer()

        # 验证服务器已初始化
        assert server is not None
        assert server.server is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_has_tools_method():
    """测试 MCP 服务器有工具方法"""
    with patch('src.mcp_server.server.load_config') as mock_config:
        mock_config.return_value = {}

        from src.mcp_server.server import RAGMCPServer

        server = RAGMCPServer()
        
        # 验证服务器有 list_tools 方法
        assert hasattr(server.server, 'list_tools')
        assert callable(server.server.list_tools)