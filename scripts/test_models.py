#!/usr/bin/env python3
"""
测试模型配置和连接

测试 LLM、Embedding 和 Reranker 模型的连接
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.rag.components.factory import create_llm, create_embedder, create_reranker
from src.rag.components import *


async def test_llm(config):
    """测试 LLM 模型"""
    print("\n" + "=" * 60)
    print("测试 LLM 模型")
    print("=" * 60)
    
    try:
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'qianfan')
        
        print(f"提供商: {provider}")
        print(f"模型: {llm_config.get(provider, {}).get('model', 'N/A')}")
        
        llm = create_llm(llm_config)
        
        print("\n正在测试 LLM 连接...")
        from src.rag.components.llms.base import LLMMessage
        response = await llm.generate([
            LLMMessage(role="user", content="你好，请用一句话介绍自己。")
        ])
        
        print(f"✅ LLM 连接成功")
        print(f"响应: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ LLM 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding(config):
    """测试 Embedding 模型"""
    print("\n" + "=" * 60)
    print("测试 Embedding 模型")
    print("=" * 60)
    
    try:
        embedding_config = config.get('embedding', {})
        provider = embedding_config.get('provider', 'siliconflow')
        
        print(f"提供商: {provider}")
        print(f"模型: {embedding_config.get(provider, {}).get('model', 'N/A')}")
        
        embedder = create_embedder(embedding_config)
        
        print("\n正在测试 Embedding 连接...")
        embeddings = embedder.get_text_embedding("这是一个测试文本")
        
        print(f"✅ Embedding 连接成功")
        print(f"向量维度: {len(embeddings)}")
        print(f"向量示例: {embeddings[:5]}")
        return True
    except Exception as e:
        print(f"❌ Embedding 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranker(config):
    """测试 Reranker 模型"""
    print("\n" + "=" * 60)
    print("测试 Reranker 模型")
    print("=" * 60)
    
    try:
        reranker_config = config.get('reranker', {})
        provider = reranker_config.get('provider', 'siliconflow')
        
        print(f"提供商: {provider}")
        print(f"模型: {reranker_config.get(provider, {}).get('model', 'N/A')}")
        
        reranker = create_reranker(reranker_config)
        
        print("\n正在测试 Reranker 连接...")
        query = "什么是人工智能？"
        documents = [
            "人工智能是计算机科学的一个分支。",
            "今天天气很好。",
            "机器学习是人工智能的核心技术。"
        ]
        
        results = await reranker.rerank(
            query=query,
            documents=documents,
            doc_ids=["1", "2", "3"],
            metadatas=[{}, {}, {}]
        )
        
        print(f"✅ Reranker 连接成功")
        print(f"重排序结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result.score:.4f}] {result.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Reranker 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("RAG-MCP-DEMO 模型配置测试")
    print("=" * 60)
    
    config = load_config()
    
    if not config:
        print("❌ 未找到配置，请检查 .env 文件")
        return
    
    import asyncio
    
    async def run_all_tests():
        return {
            "LLM": await test_llm(config),
            "Embedding": test_embedding(config),
            "Reranker": await test_reranker(config)
        }
    
    results = asyncio.run(run_all_tests())
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for model, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有模型测试通过！")
    else:
        print("\n⚠️  部分模型测试失败，请检查配置")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
