# RAG-MCP-Demo

基于 LlamaIndex 的 RAG 知识库系统，支持 MCP 协议。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/RAG-MCP-DEMO.git
cd RAG-MCP-DEMO
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API 密钥
```

### 5. 验证安装

```bash
python scripts/verify_setup.py
```

### 6. 运行 MCP Server

```bash
python -m src.mcp_server.server
```

## 功能特性

- ✅ 文档上传（支持 txt, md, pdf, docx, xlsx）
- ✅ 混合检索（向量 + BM25）
- ✅ 智能问答（千帆/SiliconFlow LLM）
- ✅ MCP 协议支持（STDIO/SSE）
- ✅ RAGAS 评估

## 文档导航

详细的文档请参阅 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| [快速开始指南](./docs/getting-started/quickstart.md) | 新手入门完整教程 |
| [安装配置指南](./docs/configuration/install.md) | 环境搭建和配置详解 |
| [架构设计文档](./docs/architecture/design.md) | 系统架构和模块设计 |
| [核心模块分析](./docs/deep-dive/core-modules.md) | 源码深度解析 |
| [MCP 协议使用指南](./docs/mcp/usage.md) | MCP 协议使用详解 |
| [扩展开发指南](./docs/development/extend.md) | 如何扩展系统功能 |
| [API 参考文档](./docs/api/reference.md) | 完整 API 参考 |
| [常见问题解答](./docs/faq/README.md) | FAQ 和问题解决 |

## 架构

- **RAG 框架**: LlamaIndex
- **向量数据库**: Chroma (本地持久化)
- **嵌入模型**: BAAI/bge-large-zh-v1.5
- **LLM**: 千帆 (ERNIE-Bot-4) / SiliconFlow / OpenAI
- **重排序**: BAAI/bge-reranker-v2-m3
- **协议**: MCP (STDIO/SSE)

## 项目结构

```
RAG-MCP-DEMO/
├── src/
│   ├── rag/              # RAG 核心模块
│   │   ├── llamaindex/   # LlamaIndex 集成
│   │   └── components/   # 可插拔组件
│   ├── mcp_server/       # MCP 服务器
│   ├── utils/            # 工具类
│   └── evaluation/       # 评估模块
├── docs/                 # 详细文档
│   ├── getting-started/  # 快速开始
│   ├── architecture/     # 架构设计
│   ├── configuration/    # 安装配置
│   ├── deep-dive/        # 源码分析
│   ├── mcp/              # MCP 协议
│   ├── development/      # 扩展开发
│   ├── api/              # API 参考
│   └── faq/              # 常见问题
├── scripts/              # 脚本工具
├── examples/             # 示例代码
├── knowledge_base/       # 知识库文档
└── tests/                # 测试用例
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit/test_config.py -v

# 验证环境
python scripts/verify_setup.py

# 测试模型
python scripts/test_models.py

# 测试完整流程
python scripts/test_full_pipeline.py
```

## 学习资源

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [MCP 协议规范](https://spec.modelcontextprotocol.io/)
- [Chroma 向量数据库](https://docs.trychroma.com/)
- [BGE 嵌入模型](https://github.com/FlagOpen/FlagEmbedding)

## 许可证

MIT