# 快速开始指南

本指南将帮助你快速搭建并运行 RAG-MCP-DEMO 项目。即使你是初学者，只需要按照下面的步骤操作即可。

## 环境要求

在开始之前，请确保你的开发环境满足以下要求：

| 要求 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.10+ | 推荐使用 3.11 或 3.12 |
| 系统 | Linux/macOS/Windows (WSL) | 推荐使用 Linux 或 macOS |
| 内存 | 4GB+ | 运行 LLM 需要较大内存 |
| 磁盘 | 10GB+ | 存储向量数据库和文档 |

## 步骤一：克隆项目

首先，从 GitHub 克隆项目到本地：

```bash
git clone https://github.com/your-repo/RAG-MCP-DEMO.git
cd RAG-MCP-DEMO
```

## 步骤二：创建虚拟环境

为了隔离项目依赖，建议使用虚拟环境：

```bash
# 使用 venv 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate

# Windows (CMD):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

激活成功后，命令行前会显示 `(venv)` 标识。

## 步骤三：安装依赖

安装项目所需的所有依赖包：

```bash
pip install -r requirements.txt
```

如果需要支持 Office 文档（Word/Excel），还需安装可选依赖：

```bash
pip install -r requirements.txt -r requirements-office.txt
# 或
pip install -e ".[office]"
```

## 步骤四：配置环境变量

项目使用环境变量来配置 API 密钥等敏感信息。

### 4.1 复制环境配置模板

```bash
cp .env.example .env
```

### 4.2 编辑 .env 文件

使用任意文本编辑器打开 `.env` 文件，填入你的 API 密钥：

```bash
# 打开 .env 文件进行编辑
nano .env  # Linux
# 或
notepad .env  # Windows
```

### 4.3 配置说明

以下是各个配置项的说明：

#### LLM（大语言模型）配置

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `qianfan`、`openai`、`siliconflow` |
| `LLM_MODEL` | 使用的模型 | `ERNIE-Bot-4`、`gpt-4` |
| `LLM_API_KEY` | LLM API 密钥 | 你的 API 密钥 |
| `LLM_BASE_URL` | API 地址（可选） | 自定义 API 端点 |

#### Embedding（向量嵌入）配置

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `EMBEDDING_PROVIDER` | 嵌入模型提供商 | `siliconflow`、`openai` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `BAAI/bge-large-zh-v1.5` |
| `EMBEDDING_API_KEY` | 嵌入模型 API 密钥 | 你的 API 密钥 |
| `EMBEDDING_BASE_URL` | API 地址（可选） | `https://api.siliconflow.cn/v1` |

#### Reranker（重排序）配置

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `RERANKER_PROVIDER` | 重排序提供商 | `siliconflow` |
| `RERANKER_MODEL` | 重排序模型 | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_API_KEY` | 重排序 API 密钥 | 你的 API 密钥 |

#### 向量数据库配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VECTOR_DB_PROVIDER` | 向量数据库类型 | `chroma` |
| `VECTOR_DB_PERSIST_DIR` | 数据库存储路径 | `./chroma_db` |

### 4.4 获取 API 密钥

本项目支持多种 LLM 提供商，你可以根据需要选择：

**SiliconFlow（推荐）**

1. 访问 [SiliconFlow](https://siliconflow.cn) 注册账号
2. 在控制台获取 API Key
3. SiliconFlow 提供免费额度，适合学习使用

**百度千帆**

1. 访问[百度智能云千帆平台](https://qianfan.baidubce.com)注册
2. 创建应用获取 API Key 和 Secret Key
3. 注意：千帆标准版需要 Secret Key，Coding Plan 版只需 API Key

## 步骤五：验证环境

运行验证脚本检查环境配置是否正确：

```bash
python scripts/verify_setup.py
```

如果所有检查都通过，你会看到类似输出：

```
============================================================
RAG-MCP-DEMO 设置验证
============================================================
1. 检查 Python 版本
Python 版本: 3.11.x
状态: 通过
...
============================================================
所有检查通过！可以开始使用了。

快速开始:
  1. 准备文档: 将文档放入 knowledge_base/ 目录
  2. 启动 MCP 服务器: python -m src.mcp_server.server
  3. 或使用示例: python examples/01_basic_ingestion.py
============================================================
```

## 步骤六：添加知识文档

将你需要用于问答的文档放入 `knowledge_base/` 目录。

项目支持以下文档格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 文本文件 | `.txt` | 纯文本文件 |
| Markdown | `.md` | Markdown 格式文档 |
| PDF | `.pdf` | PDF 文档（支持文本提取） |
| Word | `.docx` | Word 文档 |
| Excel | `.xlsx`、`.xls` | Excel 表格 |

示例：

```bash
# 查看知识库目录
ls knowledge_base/

# 添加你的文档
cp /path/to/your/document.pdf knowledge_base/
```

## 步骤七：构建索引

首次使用时，需要将文档加载到向量数据库中：

```bash
python -m src.mcp_server.server
```

或者使用示例脚本：

```bash
python examples/01_basic_ingestion.py
```

构建完成后，文档会被切分成小块并存储到 Chroma 向量数据库中。

## 步骤八：启动 MCP 服务器

现在可以启动 MCP 服务器开始使用：

```bash
python -m src.mcp_server.server
```

服务器启动后，你可以通过 MCP 客户端进行问答。

## 下一步

- 想要了解系统架构？查看 [架构设计文档](./architecture/)
- 想要深入学习源码？查看 [核心模块分析](./deep-dive/)
- 想要自定义功能？查看 [扩展开发指南](./development/)
- 遇到问题了？查看 [常见问题解答](./faq/)

## 常见问题

**Q: 运行提示缺少依赖**
A: 确保已激活虚拟环境且依赖已安装：`source venv/bin/activate && pip install -r requirements.txt`

**Q: API 密钥配置正确但无法连接**
A: 检查网络连接，确保可以访问 API 提供商的服务器

**Q: 向量数据库存储在哪里**
A: 默认存储在 `./data/chroma_db/` 目录

## 获取帮助

- 查看 [GitHub Issues](https://github.com/your-repo/RAG-MCP-DEMO/issues)
- 在项目讨论区提问
- 查看完整的 [API 参考文档](./api/)