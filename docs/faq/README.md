# 常见问题解答 (FAQ)

本文档收集了 RAG-MCP-DEMO 使用过程中常见的问题和解决方案。

## 目录

1. [安装和配置问题](#1-安装和配置问题)
2. [运行问题](#2-运行问题)
3. [API 和密钥问题](#3-api-和密钥问题)
4. [功能使用问题](#4-功能使用问题)
5. [性能问题](#5-性能问题)
6. [其他问题](#6-其他问题)

---

## 1. 安装和配置问题

### Q1.1: Python 版本要求是多少？

**问题**：运行项目时提示 Python 版本过低。

**解答**：
项目要求 Python 3.10 或更高版本。推荐使用 Python 3.11 或 3.12。

```bash
# 检查 Python 版本
python --version

# 如果版本过低，安装新版本 (Ubuntu)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Q1.2: 依赖安装失败怎么办？

**问题**：`pip install -r requirements.txt` 失败。

**解答**：

```bash
# 1. 升级 pip 和 setuptools
pip install --upgrade pip setuptools wheel

# 2. 安装编译依赖 (Ubuntu)
sudo apt install build-essential python3-dev

# 3. 如果仍然失败，尝试逐个安装
pip install llama-index
pip install chromadb
# ... 以此类推
```

### Q1.3: 虚拟环境相关问题

**问题**：不知道如何创建和激活虚拟环境。

**解答**：

```bash
# 创建虚拟环境
python -m venv venv

# 激活 (Linux/macOS)
source venv/bin/activate

# 激活 (Windows CMD)
venv\Scripts\activate.bat

# 激活 (Windows PowerShell)
venv\Scripts\Activate.ps1

# 退出虚拟环境
deactivate
```

---

## 2. 运行问题

### Q2.1: 启动 MCP 服务器报错

**问题**：运行 `python -m src.mcp_server.server` 报错。

**解答**：

1. 确保已激活虚拟环境
2. 检查 `.env` 文件配置是否正确
3. 查看错误信息确定具体问题

```bash
# 激活环境
source venv/bin/activate

# 验证环境
python scripts/verify_setup.py

# 运行服务器（查看详细错误）
python -m src.mcp_server.server
```

### Q2.2: 找不到模块

**问题**：`ModuleNotFoundError: No module named 'src'`

**解答**：

```bash
# 方法1：设置 PYTHONPATH
export PYTHONPATH=.

# 方法2：在项目根目录运行
cd /path/to/RAG-MCP-DEMO
python -m src.mcp_server.server

# 方法3：使用项目提供的启动脚本
python scripts/start_mcp_server.py
```

### Q2.3: 端口被占用

**问题**：启动 SSE 服务器时提示端口被占用。

**解答**：

```bash
# 查找占用端口的进程
lsof -i :8080  # Linux/macOS
netstat -ano | findstr :8080  # Windows

# 终止进程
kill <PID>

# 或者使用其他端口
export MCP_PORT=8081
```

---

## 3. API 和密钥问题

### Q3.1: API 密钥无效

**问题**：提示 API 密钥无效或已过期。

**解答**：

1. 登录对应的 API 提供商控制台
2. 检查 API Key 是否正确
3. 确认 API Key 是否还有余额/配额
4. 确认 API Key 是否已过期

```bash
# 检查 .env 配置
cat .env | grep API_KEY
```

### Q3.2: 无法连接到 API 服务

**问题**：网络请求超时或连接失败。

**解答**：

```bash
# 1. 检查网络连接
ping api.siliconflow.cn
ping qianfan.baidubce.com

# 2. 配置代理（如需要）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 3. 检查防火墙设置
sudo ufw status
```

### Q3.3: 免费额度用完了

**问题**：API 返回配额不足错误。

**解答**：

1. **SiliconFlow**：登录控制台充值或购买套餐
2. **百度千帆**：登录百度云控制台购买资源包
3. **OpenAI**：绑定支付方式或等待下月免费额度

---

## 4. 功能使用问题

### Q4.1: 文档上传成功但搜索不到内容

**问题**：执行 `ingest_document` 成功，但问答时找不到相关内容。

**解答**：

1. **检查文档是否被正确处理**：
```python
# 查看处理日志，确认文档数量
# 启动服务器时查看控制台输出
```

2. **检查向量数据库**：
```python
# 验证 Chroma 数据库是否包含数据
import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection("knowledge_base")
print(f"文档数量: {collection.count()}")
```

3. **检查文档格式**：
   - 确保文档是支持的格式（txt, md, pdf, docx, xlsx）
   - 确保文档编码为 UTF-8

4. **重新构建索引**：
```bash
# 删除旧索引
rm -rf ./data/chroma_db

# 重新上传文档
python -m src.mcp_server.server
# 然后调用 ingest_document
```

### Q4.2: 问答结果不准确

**问题**：回答的内容与知识库不符或不够准确。

**解答**：

1. **调整检索参数**：
```python
# 在 pipeline.py 中调整 top_k
vector_retriever = VectorIndexRetriever(
    index=self.index,
    similarity_top_k=30  # 增加候选数量
)
```

2. **调整混合检索权重**：
```python
hybrid_retriever = HybridRetriever(
    index=self.index,
    vector_retriever=vector_retriever,
    top_k=10,
    vector_weight=0.6,  # 降低向量权重
    bm25_weight=0.4     # 增加 BM25 权重
)
```

3. **添加重排序模型**：
   - 在 `.env` 中配置 Reranker
   - 重排序可以显著提升结果准确性

4. **优化文档内容**：
   - 文档内容越清晰、结构化，回答越准确
   - 避免过短的文档片段

### Q4.3: 支持哪些文档格式？

**问题**：上传的文档格式不支持。

**解答**：

当前支持的文档格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 纯文本 | `.txt` | 简单文本文件 |
| Markdown | `.md` | Markdown 格式 |
| PDF | `.pdf` | PDF 文档（需能提取文本） |
| Word | `.docx` | Word 文档 |
| Excel | `.xlsx`, `.xls` | Excel 表格 |

如需支持更多格式，可以在 `src/rag/components/loaders/` 中添加新的 Loader。

---

## 5. 性能问题

### Q5.1: 索引构建很慢

**问题**：处理大量文档时，构建索引需要很长时间。

**解答**：

1. **批量处理**：避免一次处理太少文档
2. **使用更快的嵌入模型**：如 `bge-small-zh-v1.5`
3. **减少分块大小**：可以加快处理速度
4. **使用 SSD**：向量数据库存储在 SSD 上会更快

```python
# 调整分块大小
processor = SmartTextProcessor(doc_type='general')
processor.CHUNK_SIZES = {'general': (100, 300)}  # 更小的块
```

### Q5.2: 问答响应慢

**问题**：每次提问都需要等待很长时间。

**解答**：

1. **使用缓存**：首次问答后，系统会缓存结果
2. **减少 `top_k`**：减少需要处理的文档数量
3. **选择更快的 LLM**：如使用 qianfan-turbo
4. **优化网络**：使用国内 API 服务商

### Q5.3: 内存占用过高

**问题**：运行一段时间后内存占用越来越高。

**解答**：

1. **定期重启服务**：清理内存缓存
2. **减少会话内存**：
```python
memory=ChatMemoryBuffer.from_defaults(token_limit=1000)  # 减少到 1000 tokens
```
3. **使用更小的模型**：如选择 `bge-small` 而非 `bge-large`

---

## 6. 其他问题

### Q6.1: 如何查看日志？

**解答**：

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 运行服务器，日志会输出到控制台
python -m src.mcp_server.server
```

或者查看日志文件（如果配置了文件日志）。

### Q6.2: 如何更新项目？

**解答**：

```bash
# 进入项目目录
cd RAG-MCP-DEMO

# 拉取最新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt

# 如果有数据库迁移，需要运行迁移脚本
python scripts/migrate.py
```

### Q6.3: 如何贡献代码？

**解答**：

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 开发并测试
4. 提交 Pull Request

### Q6.4: 项目使用的许可证是什么？

**解答**：MIT 许可证，允许自由使用和修改。

### Q6.5: 如何获取帮助？

**解答**：

1. 查看文档：
   - [快速开始](../getting-started/quickstart/)
   - [架构设计](../architecture/design/)
   - [核心模块分析](../deep-dive/)

2. 查看 GitHub Issues

3. 在项目讨论区提问

---

## 报告新问题

如果遇到的问题不在上述列表中，请：

1. 先查看相关文档
2. 搜索 GitHub Issues 是否已有类似问题
3. 如果没有，提 Issue 并包含以下信息：
   - 操作系统和 Python 版本
   - 完整的错误信息
   - 复现步骤
   - 已尝试的解决方案