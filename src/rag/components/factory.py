"""
组件工厂 - 统一的组件创建接口

使用注册表模式，根据配置动态创建组件实例
"""

from typing import Any, Dict
from src.utils.registry import Registry


def create_llm(config: Dict[str, Any]):
    """创建 LLM 组件
    
    【参数】
        config: LLM 配置字典，包含 provider 和对应供应商的配置
    
    【返回】
        LLM 实例
    
    【示例】
        config = {
            "provider": "openai",
            "openai": {
                "api_key": "sk-xxx",
                "model": "gpt-4"
            }
        }
        llm = create_llm(config)
    """
    provider = config.get("provider")
    if not provider:
        raise ValueError("LLM 配置缺少 provider 字段")
    
    provider_config = config.get(provider, {})
    if not provider_config:
        raise ValueError(f"LLM 配置缺少 {provider} 供应商的配置")
    
    return Registry.create("llm", provider, provider_config)


def create_embedder(config: Dict[str, Any]):
    """创建 Embedder 组件
    
    【参数】
        config: Embedding 配置字典
    
    【返回】
        Embedder 实例
    """
    provider = config.get("provider")
    if not provider:
        raise ValueError("Embedding 配置缺少 provider 字段")
    
    provider_config = config.get(provider, {})
    if not provider_config:
        raise ValueError(f"Embedding 配置缺少 {provider} 供应商的配置")
    
    return Registry.create("embedder", provider, provider_config)


def create_reranker(config: Dict[str, Any]):
    """创建 Reranker 组件
    
    【参数】
        config: Reranker 配置字典
    
    【返回】
        Reranker 实例
    """
    provider = config.get("provider")
    if not provider:
        raise ValueError("Reranker 配置缺少 provider 字段")
    
    provider_config = config.get(provider, {})
    if not provider_config:
        raise ValueError(f"Reranker 配置缺少 {provider} 供应商的配置")
    
    return Registry.create("reranker", provider, provider_config)


def create_vector_store(config: Dict[str, Any]):
    """创建 Vector Store 组件
    
    【参数】
        config: Vector Store 配置字典
    
    【返回】
        VectorStore 实例
    """
    provider = config.get("provider", "chroma")
    provider_config = config.get(provider, {})
    
    return Registry.create("vector_store", provider, provider_config)


def list_available_components():
    """列出所有可用的组件"""
    return {
        "llm": Registry.list_components("llm"),
        "embedder": Registry.list_components("embedder"),
        "reranker": Registry.list_components("reranker"),
        "vector_store": Registry.list_components("vector_store")
    }
