#!/usr/bin/env python3
"""
设置验证脚本 - 检查环境配置和依赖安装

使用方法:
    python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本")
    print("=" * 60)
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        print("状态: 通过")
        return True
    else:
        print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        print("状态: 失败 - 需要 Python 3.9+")
        return False


def check_dependencies():
    """检查核心依赖是否安装"""
    print("\n" + "=" * 60)
    print("2. 检查核心依赖")
    print("=" * 60)

    dependencies = [
        ("llama_index", "llama-index"),
        ("chromadb", "chromadb"),
        ("qianfan", "qianfan"),
        ("httpx", "httpx"),
        ("pydantic", "pydantic"),
        ("yaml", "pyyaml"),
    ]

    all_passed = True
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"  {package}: 已安装")
        except ImportError:
            print(f"  {package}: 未安装")
            all_passed = False

    if all_passed:
        print("状态: 通过")
    else:
        print("状态: 失败 - 请运行: pip install -r requirements.txt")

    return all_passed


def check_env_file():
    """检查环境变量文件"""
    print("\n" + "=" * 60)
    print("3. 检查环境变量配置")
    print("=" * 60)

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        print("状态: 警告 - .env 文件不存在")
        if env_example.exists():
            print(f"提示: 复制 {env_example} 为 .env 并填写配置")
        return False

    # 读取 .env 文件
    with open(env_file) as f:
        content = f.read()

    # 检查必需的配置（新的统一命名格式）
    required_vars = [
        "LLM_API_KEY",
        "LLM_SECRET_KEY",
        "EMBEDDING_API_KEY",
        "RERANKER_API_KEY",
    ]

    missing = []
    for var in required_vars:
        if var not in content or f"{var}=your-" in content:
            missing.append(var)

    if missing:
        print("状态: 警告 - 以下配置项需要设置:")
        for var in missing:
            print(f"  - {var}")
        return False
    else:
        print("状态: 通过")
        return True


def check_project_structure():
    """检查项目结构"""
    print("\n" + "=" * 60)
    print("4. 检查项目结构")
    print("=" * 60)

    required_dirs = [
        "src",
        "src/rag",
        "src/mcp_server",
        "src/utils",
        "config",
        "examples",
        "tests",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  {dir_path}/: 存在")
        else:
            print(f"  {dir_path}/: 不存在")
            all_exist = False

    if all_exist:
        print("状态: 通过")
    else:
        print("状态: 失败 - 项目结构不完整")

    return all_exist


def check_imports():
    """检查项目模块是否可以导入"""
    print("\n" + "=" * 60)
    print("5. 检查项目模块导入")
    print("=" * 60)

    # 将项目根目录添加到路径（使 src 包可导入）
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    modules = [
        "src.utils.config",
        "src.utils.logger",
        "src.utils.registry",
        "src.rag.components.llms.base",
        "src.rag.components.llms.qianfan_llm",
        "src.rag.components.embedders.base",
        "src.rag.components.embedders.siliconflow_embedder",
        "src.rag.components.vector_stores.chroma_store",
        "src.rag.components.rerankers.siliconflow_reranker",
        "src.rag.llamaindex.hybrid_retriever",
    ]

    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"  {module}: 导入成功")
        except Exception as e:
            print(f"  {module}: 导入失败 - {e}")
            all_passed = False

    if all_passed:
        print("状态: 通过")
    else:
        print("状态: 失败 - 部分模块导入出错")

    return all_passed


def main():
    """主函数"""
    print("RAG-MCP-DEMO 设置验证")
    print("=" * 60)

    results = [
        ("Python 版本", check_python_version()),
        ("核心依赖", check_dependencies()),
        ("环境变量", check_env_file()),
        ("项目结构", check_project_structure()),
        ("模块导入", check_imports()),
    ]

    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("所有检查通过！可以开始使用了。")
        print("\n快速开始:")
        print("  1. 准备文档: 将文档放入 knowledge_base/ 目录")
        print("  2. 启动 MCP 服务器: python -m src.mcp_server.server")
        print("  3. 或使用示例: python examples/01_basic_ingestion.py")
    else:
        print("部分检查未通过，请根据提示修复问题。")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
