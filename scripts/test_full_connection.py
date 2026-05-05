#!/usr/bin/env python3
"""
全面测试模型联通性

测试：
1. API 连接
2. LLM 功能
3. Embedding 功能
4. Reranker 功能
"""

import httpx
import json
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.rag.components.factory import create_llm, create_embedder, create_reranker
from src.rag.components import *
from src.rag.components.llms.base import LLMMessage


def test_api_connection():
    """测试原始 API 连接"""
    print("\n" + "=" * 60)
    print("1. 测试原始 API 连接")
    print("=" * 60)
    
    config = load_config()
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'myapi')
    provider_config = llm_config.get(provider, {})
    
    base_url = provider_config.get('base_url', '')
    api_key = provider_config.get('api_key', '')
    model = provider_config.get('model', '')
    
    if not base_url or not api_key:
        print("❌ 缺少必要的配置")
        return False
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    results = {}
    
    # 测试 LLM API
    print("\n测试 LLM API...")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "你好"}]
                }
            )
            
            if response.status_code == 200:
                text = response.text.strip()
                if text:
                    data = json.loads(text)
                    if 'choices' in data:
                        print("✅ LLM API 连接正常")
                        print(f"   响应示例: {data['choices'][0]['message']['content'][:50]}...")
                        results['llm'] = True
                    else:
                        print(f"❌ LLM API 响应格式异常: {text[:100]}")
                        results['llm'] = False
                else:
                    print("❌ LLM API 返回空响应")
                    results['llm'] = False
            else:
                print(f"❌ LLM API 失败: {response.status_code}")
                results['llm'] = False
    except Exception as e:
        print(f"❌ LLM API 错误: {e}")
        results['llm'] = False
    
    # 测试 Embedding API
    print("\n测试 Embedding API...")
    embedding_config = config.get('embedding', {}).get(config.get('embedding', {}).get('provider', 'myapi'), {})
    embed_model = embedding_config.get('model', 'BAAI/bge-m3')
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{base_url}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": embed_model,
                    "input": "测试文本"
                }
            )
            
            if response.status_code == 200:
                text = response.text.strip()
                if text:
                    data = json.loads(text)
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0]['embedding']
                        print("✅ Embedding API 连接正常")
                        print(f"   向量维度: {len(embedding)}")
                        results['embedding'] = True
                    else:
                        print(f"❌ Embedding API 响应格式异常")
                        results['embedding'] = False
                else:
                    print("❌ Embedding API 返回空响应")
                    results['embedding'] = False
            else:
                print(f"❌ Embedding API 失败: {response.status_code}")
                results['embedding'] = False
    except Exception as e:
        print(f"❌ Embedding API 错误: {e}")
        results['embedding'] = False
    
    # 测试 Reranker API
    print("\n测试 Reranker API...")
    reranker_config = config.get('reranker', {}).get(config.get('reranker', {}).get('provider', 'myapi'), {})
    rerank_model = reranker_config.get('model', 'BAAI/bge-reranker-v2-m3')
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{base_url}/v1/rerank",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": rerank_model,
                    "query": "什么是AI？",
                    "documents": ["AI是人工智能", "今天天气好"]
                }
            )
            
            if response.status_code == 200:
                text = response.text.strip()
                if text:
                    data = json.loads(text)
                    if 'results' in data:
                        print("✅ Reranker API 连接正常")
                        print(f"   结果数量: {len(data['results'])}")
                        results['reranker'] = True
                    else:
                        print(f"❌ Reranker API 响应格式异常")
                        results['reranker'] = False
                else:
                    print("❌ Reranker API 返回空响应")
                    results['reranker'] = False
            else:
                print(f"❌ Reranker API 失败: {response.status_code}")
                results['reranker'] = False
    except Exception as e:
        print(f"❌ Reranker API 错误: {e}")
        results['reranker'] = False
    
    return all(results.values())


async def test_llm_function():
    """测试 LLM 功能"""
    print("\n" + "=" * 60)
    print("2. 测试 LLM 功能")
    print("=" * 60)
    
    try:
        config = load_config()
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'myapi')
        
        print(f"提供商: {provider}")
        print(f"模型: {llm_config.get(provider, {}).get('model', 'N/A')}")
        
        llm = create_llm(llm_config)
        
        print("\n正在生成回答...")
        response = await llm.generate([
            LLMMessage(role="user", content="请用一句话介绍什么是RAG技术？")
        ])
        
        print(f"✅ LLM 功能正常")
        print(f"回答: {response.content}")
        return True
    except Exception as e:
        print(f"❌ LLM 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_function():
    """测试 Embedding 功能"""
    print("\n" + "=" * 60)
    print("3. 测试 Embedding 功能")
    print("=" * 60)
    
    try:
        config = load_config()
        embedding_config = config.get('embedding', {})
        provider = embedding_config.get('provider', 'myapi')
        
        print(f"提供商: {provider}")
        print(f"模型: {embedding_config.get(provider, {}).get('model', 'N/A')}")
        
        embedder = create_embedder(embedding_config)
        
        print("\n正在生成向量...")
        text = "这是一个测试文本，用于验证嵌入模型的功能。"
        embedding = embedder.get_text_embedding(text)
        
        print(f"✅ Embedding 功能正常")
        print(f"向量维度: {len(embedding)}")
        print(f"向量示例: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"❌ Embedding 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranker_function():
    """测试 Reranker 功能"""
    print("\n" + "=" * 60)
    print("4. 测试 Reranker 功能")
    print("=" * 60)
    
    try:
        config = load_config()
        reranker_config = config.get('reranker', {})
        provider = reranker_config.get('provider', 'myapi')
        
        print(f"提供商: {provider}")
        print(f"模型: {reranker_config.get(provider, {}).get('model', 'N/A')}")
        
        reranker = create_reranker(reranker_config)
        
        print("\n正在进行重排序...")
        query = "什么是机器学习？"
        documents = [
            "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            "今天天气晴朗，适合外出游玩。",
            "深度学习是机器学习的一种方法，使用神经网络进行学习。"
        ]
        
        results = await reranker.rerank(
            query=query,
            documents=documents,
            doc_ids=["1", "2", "3"],
            metadatas=[{}, {}, {}]
        )
        
        print(f"✅ Reranker 功能正常")
        print(f"重排序结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. [相关度: {result.score:.4f}] {result.text[:60]}...")
        return True
    except Exception as e:
        print(f"❌ Reranker 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("RAG-MCP-DEMO 模型联通性全面测试")
    print("=" * 60)
    
    results = {}
    
    # 1. 测试原始 API 连接
    results['api'] = test_api_connection()
    
    # 2. 测试 LLM 功能
    results['llm'] = asyncio.run(test_llm_function())
    
    # 3. 测试 Embedding 功能
    results['embedding'] = test_embedding_function()
    
    # 4. 测试 Reranker 功能
    results['reranker'] = asyncio.run(test_reranker_function())
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！模型联通性良好。")
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
