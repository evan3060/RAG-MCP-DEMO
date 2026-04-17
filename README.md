# RAG-MCP-Demo

基于 LlamaIndex 的 RAG 知识库系统，支持 MCP 协议

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python scripts/verify_setup.py
```

### 3. 配置环境变量

```bash
export QIANFAN_API_KEY="your-key"
export QIANFAN_SECRET_KEY="your-secret"
export SILICONFLOW_API_KEY="your-key"
```

### 3. 运行 MCP Server

```bash
# STDIO 模式
python -m src.mcp_server.server
```

## 功能特性

- ✅ 文档上传（支持 txt, md, pdf, docx, xlsx）
- ✅ 混合检索（向量 + BM25）
- ✅ 智能问答（千帆 LLM）
- ✅ MCP 协议支持
- ✅ RAGAS 评估

## 架构

- **RAG 框架**: LlamaIndex
- **向量数据库**: Chroma (本地)
- **嵌入模型**: SiliconFlow
- **LLM**: 千帆 (ERNIE-Bot-4)
- **协议**: MCP (STDIO/SSE)

## 开发

```bash
# 运行测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit/test_config.py -v
```

## 许可证

MIT
