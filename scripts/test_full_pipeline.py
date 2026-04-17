#!/usr/bin/env python3
"""
完整流程测试 - 文档上传 + 问答

使用方法:
    python scripts/test_full_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 启用嵌套事件循环
import nest_asyncio
nest_asyncio.apply()

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.utils.config import load_config
from src.rag.llamaindex.pipeline import RAGPipeline


async def test_full_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("完整流程测试 - 文档上传 + 问答")
    print("=" * 60)

    # 加载配置
    config = load_config()
    print(f"\n配置加载成功")
    print(f"  LLM Provider: {config.get('llm', {}).get('provider')}")
    print(f"  Embedding Model: {config.get('embedding', {}).get('siliconflow', {}).get('model')}")

    # 创建 Pipeline
    pipeline = RAGPipeline(config)

    # 1. 构建索引
    print("\n" + "=" * 60)
    print("1. 构建知识库索引")
    print("=" * 60)
    kb_path = Path(__file__).parent.parent / "knowledge_base"
    await pipeline.build_index(str(kb_path))
    print("✅ 索引构建完成")

    # 2. 测试检索
    print("\n" + "=" * 60)
    print("2. 测试检索")
    print("=" * 60)
    query = "人工智能的发展历程"
    print(f"查询: {query}")

    search_results = await pipeline.search(query, top_k=3)
    print(f"\n检索结果 ({len(search_results)} 条):")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. [相似度: {result['score']:.4f}]")
        print(f"     {result['content'][:100]}...")

    # 3. 测试问答
    print("\n" + "=" * 60)
    print("3. 测试问答")
    print("=" * 60)
    question = "人工智能有哪些主要应用领域？"
    print(f"问题: {question}\n")

    result = await pipeline.ask(question)

    print(f"回答:\n{result['answer']}\n")
    print(f"参考来源 ({len(result['sources'])} 条):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. [相似度: {source['score']:.4f}]")
        print(f"     {source['content'][:80]}...")

    print("\n" + "=" * 60)
    print("✅ 完整流程测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_full_pipeline())
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
