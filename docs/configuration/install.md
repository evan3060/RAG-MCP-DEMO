# 安装配置指南

本文档详细介绍 RAG-MCP-DEMO 的安装和配置过程，帮助开发者从零开始搭建开发环境。

## 环境准备

### 系统要求

| 要求项 | 最低配置 | 推荐配置 |
|--------|----------|----------|
| 操作系统 | Linux/macOS/Windows (WSL) | Linux (Ubuntu 20.04+) |
| Python 版本 | 3.10+ | 3.11 或 3.12 |
| 内存 | 4 GB | 8 GB 以上 |
| 磁盘空间 | 10 GB | 20 GB 以上 |
| 网络 | 可访问国内 API | 可访问国际 API |

### 推荐开发环境

- **操作系统**：Ubuntu 20.04+ / macOS 12+
- **终端**：iTerm2 (macOS) / GNOME Terminal (Linux)
- **编辑器**：VS Code / PyCharm

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/RAG-MCP-DEMO.git
cd RAG-MCP-DEMO
```

### 2. 创建虚拟环境

Python 虚拟环境可以隔离项目依赖，避免与其他项目冲突。

```bash
# 使用 Python 内置的 venv 模块创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate

# 验证激活成功
# 命令行前应显示 (venv)
```

> **提示**：如果你使用 `conda`，也可以使用 `conda create -n rag-mcp python=3.11` 创建环境。

### 3. 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 可选：安装 Office 文档支持
pip install python-docx openpyxl xlrd
```

### 4. 配置文件

#### 4.1 环境变量文件

项目使用 `.env` 文件存储配置：

```bash
# 复制模板文件
cp .env.example .env

# 编辑配置文件
nano .env  # Linux
# 或
notepad .env  # Windows
```

#### 4.2 配置文件详解

以下是完整的配置项说明：

```bash
# ============================================================
# LLM (大语言模型) 配置
# ============================================================

# LLM 提供商：qianfan (百度千帆) / openai / siliconflow
LLM_PROVIDER=qianfan

# 模型名称
LLM_MODEL=ERNIE-Bot-4

# API 密钥（必需）
LLM_API_KEY=your-api-key-here

# Secret Key（仅千帆标准版需要）
# LLM_SECRET_KEY=your-secret-key

# 自定义 API 地址（可选）
# LLM_BASE_URL=https://qianfan.baidubce.com/v2/coding

# ============================================================
# Embedding (向量嵌入) 配置
# ============================================================

# 嵌入模型提供商：siliconflow / openai
EMBEDDING_PROVIDER=siliconflow

# 嵌入模型名称
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 嵌入模型 API 密钥（必需）
EMBEDDING_API_KEY=your-embedding-api-key

# 嵌入模型 API 地址（可选）
# EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1

# ============================================================
# Reranker (重排序) 配置
# ============================================================

# 重排序提供商：siliconflow
RERANKER_PROVIDER=siliconflow

# 重排序模型名称
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# 重排序 API 密钥（必需）
RERANKER_API_KEY=your-reranker-api-key

# ============================================================
# 向量数据库配置
# ============================================================

# 向量数据库类型：chroma
VECTOR_DB_PROVIDER=chroma

# 数据库持久化路径
VECTOR_DB_PERSIST_DIR=./data/chroma_db

# ============================================================
# 应用配置（可选）
# ============================================================

# 日志级别：DEBUG / INFO / WARNING / ERROR
LOG_LEVEL=INFO

# 知识库目录
KNOWLEDGE_BASE_DIR=./knowledge_base

# MCP 服务器传输模式：stdio / sse
MCP_TRANSPORT=stdio

# SSE 模式端口
MCP_PORT=8080
```

## API 密钥获取

### 方案一：SiliconFlow（推荐）

SiliconFlow 是一个提供 AI 模型 API 的平台，在国内访问速度快，价格实惠。

1. 访问 [SiliconFlow 官网](https://siliconflow.cn)
2. 注册账号并完成实名认证
3. 进入控制台 → API 密钥
4. 创建新的 API Key
5. 复制密钥并填入配置

**免费额度**：
- 注册赠送积分
- 嵌入模型按字符计费
- 重排序模型按次计费

### 方案二：百度千帆

百度千帆是百度智能云提供的大模型平台。

1. 访问 [百度千帆平台](https://qianfan.baidubce.com)
2. 注册百度账号并完成企业认证
3. 创建应用：
   - 进入「应用管理」→「创建应用」
   - 选择「千帆大模型平台」
   - 记下 API Key 和 Secret Key
4. 在控制台开通需要的模型服务

**配置示例**：

```bash
# 千帆标准版
LLM_PROVIDER=qianfan
LLM_MODEL=ERNIE-Bot-4
LLM_API_KEY=your-api-key
LLM_SECRET_KEY=your-secret-key

# 千帆 Coding Plan 版（推荐）
LLM_PROVIDER=qianfan
LLM_MODEL=ernie-4.5-8k-preview  # 或其他 Coding Plan 模型
LLM_API_KEY=your-coding-plan-key
# 不需要 Secret Key
LLM_BASE_URL=https://qianfan.baidubce.com/v2/coding
```

### 方案三：OpenAI

如果你有国际网络访问能力，可以使用 OpenAI。

1. 访问 [OpenAI Platform](https://platform.openai.com)
2. 注册账号并绑定支付方式
3. 创建 API Key：Settings → API keys
4. 充值或使用免费额度

**配置示例**：

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxx
```

## 验证安装

### 运行验证脚本

```bash
python scripts/verify_setup.py
```

预期输出：

```
============================================================
RAG-MCP-DEMO 设置验证
============================================================
1. 检查 Python 版本
Python 版本: 3.11.x
状态: 通过

2. 检查核心依赖
  llama-index: 已安装
  chromadb: 已安装
  qianfan: 已安装
  httpx: 已安装
  pydantic: 已安装
  pyyaml: 已安装
状态: 通过

3. 检查环境变量配置
状态: 通过

4. 检查项目结构
  src/: 存在
  src/rag/: 存在
  ...
状态: 通过

5. 检查项目模块导入
  src.utils.config: 导入成功
  src.rag.llamaindex.pipeline: 导入成功
  ...
状态: 通过

============================================================
所有检查通过！可以开始使用了。
============================================================
```

### 常见验证失败及解决方案

#### 问题 1：Python 版本过低

```
错误：Python 版本: 3.9.x
状态: 失败 - 需要 Python 3.10+
```

**解决方案**：
```bash
# 安装 Python 3.11 (Ubuntu)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### 问题 2：依赖安装失败

```
错误：ERROR: Could not build wheels for xxx
```

**解决方案**：
```bash
# 安装编译依赖 (Ubuntu)
sudo apt install build-essential python3-dev

# 升级 pip
pip install --upgrade pip setuptools wheel
```

#### 问题 3：API 密钥未配置

```
状态: 警告 - 以下配置项需要设置:
  - LLM_API_KEY
  - EMBEDDING_API_KEY
```

**解决方案**：确保 `.env` 文件中的 API 密钥已正确填写，且不是占位符 `your-xxx`。

## 目录结构说明

```
RAG-MCP-DEMO/
├── src/                      # 源代码目录
│   ├── rag/                  # RAG 核心模块
│   │   ├── llamaindex/       # LlamaIndex 集成
│   │   │   ├── pipeline.py   # RAG 流程
│   │   │   └── hybrid_retriever.py  # 混合检索
│   │   └── components/       # 可插拔组件
│   │       ├── llms/         # LLM 实现
│   │       ├── embedders/    # 嵌入模型
│   │       ├── rerankers/    # 重排序
│   │       ├── vector_stores/ # 向量存储
│   │       └── loaders/      # 文档加载器
│   ├── mcp_server/           # MCP 服务器
│   ├── utils/                # 工具类
│   └── evaluation/           # 评估模块
├── docs/                     # 文档目录
├── scripts/                  # 脚本目录
│   ├── verify_setup.py       # 环境验证
│   ├── test_models.py        # 模型测试
│   ├── test_full_pipeline.py # 流程测试
│   └── evaluate_with_ground_truth.py  # 评估
├── examples/                 # 示例代码
├── knowledge_base/          # 知识库文档目录
├── data/                    # 数据存储目录
│   └── chroma_db/           # Chroma 数据库
├── tests/                   # 测试目录
├── requirements.txt         # 依赖列表
├── pyproject.toml          # 项目配置
├── .env.example            # 环境变量模板
└── README.md               # 项目说明
```

## 高级配置

### 使用代理

如果网络无法直接访问某些 API 服务，可以配置代理：

```bash
# HTTP 代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 或者在代码中设置（使用 httpx）
import httpx

client = httpx.Client(
    proxies={
        "http://": "http://127.0.0.1:7890",
        "https://": "http://127.0.0.1:7890"
    }
)
```

### 使用自定义模型

项目支持使用自定义的嵌入模型或 LLM。只需修改配置：

```bash
# 自定义嵌入模型
EMBEDDING_PROVIDER=siliconflow
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5  # 使用基础版本

# 自定义 LLM
LLM_PROVIDER=siliconflow
LLM_MODEL=Qwen/Qwen2-7B-Instruct  # 使用 Qwen 模型
```

### 多知识库配置

项目支持配置多个知识库目录：

```bash
# 知识库目录（可多个，用逗号分隔）
KNOWLEDGE_BASE_DIR=./knowledge_base,./docs
```

### 日志配置

```bash
# 日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
LOG_LEVEL=DEBUG  # 开发时使用详细日志
```

## 下一步

- 环境搭建完成？开始 [快速开始](../getting-started/quickstart/)
- 想要了解系统架构？查看 [架构设计文档](../architecture/design/)
- 遇到问题了？查看 [常见问题解答](../faq/)