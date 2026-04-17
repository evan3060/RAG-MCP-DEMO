#!/usr/bin/env python3
"""
模型配置测试脚本

测试内容：
1. 加载配置
2. 测试 Embedding 模型
3. 测试 LLM 模型
4. 测试 Reranker 模型

使用方法:
    python scripts/test_models.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.utils.config import load_config
from src.rag.components.embedders.siliconflow_embedder import SiliconFlowEmbedder
from src.rag.components.llms.siliconflow_llm import SiliconFlowLLM, LLMMessage
from src.rag.components.rerankers.siliconflow_reranker import SiliconFlowReranker


async def test_embedding(config):
    """测试 Embedding 模型"""
    print("\n" + "=" * 60)
    print("1. 测试 Embedding 模型")
    print("=" * 60)

    embedding_config = config.get("embedding", {}).get("siliconflow", {})
    print(f"模型: {embedding_config.get('model')}")
    print(f"Base URL: {embedding_config.get('base_url')}")

    try:
        embedder = SiliconFlowEmbedder({
            "api_key": embedding_config.get("api_key"),
            "model": embedding_config.get("model"),
            "base_url": embedding_config.get("base_url")
        })

        # 测试单条文本
        test_text = "这是一个测试文本"
        print(f"\n测试文本: '{test_text}'")

        embedding = await embedder.embed(test_text)
        print(f"嵌入维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")
        print("状态: 通过")
        return True

    except Exception as e:
        print(f"状态: 失败 - {e}")
        return False


async def test_llm(config):
    """测试 LLM 模型"""
    print("\n" + "=" * 60)
    print("2. 测试 LLM 模型")
    print("=" * 60)

    llm_config = config.get("llm", {}).get("siliconflow", {})
    print(f"模型: {llm_config.get('model')}")
    print(f"Base URL: {llm_config.get('base_url')}")

    try:
        llm = SiliconFlowLLM({
            "api_key": llm_config.get("api_key"),
            "model": llm_config.get("model"),
            "base_url": llm_config.get("base_url"),
            "temperature": 0.7
        })

        messages = [
            LLMMessage(role="system", content="你是一个有用的助手。"),
            LLMMessage(role="user", content="你好，请用一句话介绍自己。")
        ]

        print("\n发送消息...")
        response = await llm.generate(messages)

        print(f"模型回答: {response.content[:200]}...")
        print(f"Token 使用: {response.usage}")
        print("状态: 通过")
        return True

    except Exception as e:
        print(f"状态: 失败 - {e}")
        return False


async def test_reranker(config):
    """测试 Reranker 模型"""
    print("\n" + "=" * 60)
    print("3. 测试 Reranker 模型")
    print("=" * 60)

    reranker_config = config.get("reranker", {}).get("siliconflow", {})
    print(f"模型: {reranker_config.get('model')}")
    print(f"Base URL: {reranker_config.get('base_url')}")

    try:
        reranker = SiliconFlowReranker({
            "api_key": reranker_config.get("api_key"),
            "model": reranker_config.get("model"),
            "base_url": reranker_config.get("base_url"),
            "top_n": 3
        })

        query = "什么是人工智能？"
        documents = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个子领域。",
            "深度学习使用神经网络进行学习。",
            "今天的天气很好。",
            "Python 是一种编程语言。"
        ]
        doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        metadatas = [{"source": f"doc{i+1}"} for i in range(5)]

        print(f"\n查询: '{query}'")
        print(f"文档数: {len(documents)}")

        results = await reranker.rerank(query, documents, doc_ids, metadatas)

        print("\n重排序结果 (Top 3):")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. [{result.score:.4f}] {result.text[:50]}...")

        print("状态: 通过")
        return True

    except Exception as e:
        print(f"状态: 失败 - {e}")
        return False


async def main():
    """主函数"""
    print("=" * 60)
    print("模型配置测试")
    print("=" * 60)

    # 加载配置
    try:
        config = load_config("config/default.yaml")
        print("\n配置加载成功")
        print(f"LLM Provider: {config.get('llm', {}).get('provider')}")
        print(f"Embedding Provider: {config.get('embedding', {}).get('provider')}")
        print(f"Reranker Provider: {config.get('reranker', {}).get('provider')}")
    except Exception as e:
        print(f"配置加载失败: {e}")
        return 1

    # 运行测试
    results = []

    results.append(("Embedding", await test_embedding(config)))
    results.append(("LLM", await test_llm(config)))
    results.append(("Reranker", await test_reranker(config)))

    # 汇总
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n所有测试通过！")
        return 0
    else:
        print("\n部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
