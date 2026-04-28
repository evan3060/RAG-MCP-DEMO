# AGENTS.md

## 项目概述

基于 LlamaIndex 的 RAG 知识库系统，支持 MCP 协议。

## 快速命令

```bash
# 验证环境
python scripts/verify_setup.py

# 启动 MCP Server
python -m src.mcp_server.server

# 运行测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行特定测试
pytest tests/unit/test_config.py -v

# 运行测试并生成覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing
```

## 环境配置

必需的环境变量（见 `.env.example`）：
- `LLM_API_KEY` - LLM API 密钥
- `EMBEDDING_API_KEY` - 嵌入模型 API 密钥  
- `RERANKER_API_KEY` - 重排序模型 API 密钥

可选：
- `LLM_PROVIDER=qianfan|openai|siliconflow`
- `EMBEDDING_PROVIDER=siliconflow|openai`
- `VECTOR_DB_PERSIST_DIR=./chroma_db`

## 项目结构

- `src/rag/` - RAG 核心模块（LlamaIndex pipeline, hybrid retriever）
- `src/mcp_server/` - MCP 协议服务器
- `src/utils/` - 工具类（config, logger, registry）
- `src/evaluation/` - RAGAS 评估模块
- `scripts/` - 脚本（验证、测试、评估）
- `examples/` - 使用示例
- `tests/` - 测试代码
  - `tests/unit/` - 单元测试
  - `tests/integration/` - 集成测试

## 技术栈

- RAG: LlamaIndex
- 向量库: Chroma
- LLM: 千帆 (ERNIE-Bot-4) / OpenAI / SiliconFlow
- Embedding: BAAI/bge-large-zh-v1.5
- Reranker: BAAI/bge-reranker-v2-m3

## 注意事项

- 使用 venv 虚拟环境（`venv/bin/python`）
- MCP 支持 STDIO 和 SSE 两种传输模式
- 文档支持：txt, md, pdf, docx, xlsx

## 测试标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试  
- `@pytest.mark.slow` - 耗时测试
- `@pytest.mark.requires_api` - 需要 API 密钥的测试