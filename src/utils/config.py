"""
配置管理模块 - 加载和管理应用配置

支持两种配置方式：
1. YAML 配置文件 (config/default.yaml)
2. 环境变量 (.env)
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def _load_env_file():
    """加载 .env 文件"""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def _expand_env_vars(value: Any) -> Any:
    """递归展开配置中的环境变量

    示例:
        "${API_KEY}" -> "sk-xxx" (从环境变量读取)
        "${UNDEFINED:-default}" -> "default" (使用默认值)
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'

        def replace(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, match.group(0))

        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _load_from_env() -> Dict[str, Any]:
    """从环境变量加载配置"""
    # LLM 配置
    llm_config = {}
    if os.getenv("LLM_PROVIDER"):
        llm_config["provider"] = os.getenv("LLM_PROVIDER")
        llm_config[llm_config["provider"]] = {
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": os.getenv("LLM_MODEL", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7"))
        }
    
    # Embedding 配置
    embedding_config = {}
    if os.getenv("EMBEDDING_PROVIDER"):
        embedding_config["provider"] = os.getenv("EMBEDDING_PROVIDER")
        embedding_config[embedding_config["provider"]] = {
            "api_key": os.getenv("EMBEDDING_API_KEY", ""),
            "model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
            "base_url": os.getenv("EMBEDDING_BASE_URL", "")
        }
    
    # Reranker 配置
    reranker_config = {}
    if os.getenv("RERANKER_PROVIDER"):
        reranker_config["provider"] = os.getenv("RERANKER_PROVIDER")
        reranker_config[reranker_config["provider"]] = {
            "api_key": os.getenv("RERANKER_API_KEY", ""),
            "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            "base_url": os.getenv("RERANKER_BASE_URL", "")
        }
    
    # Vector Store 配置
    vector_store_config = {
        "provider": os.getenv("VECTOR_DB_PROVIDER", "chroma"),
        "persist_dir": os.getenv("VECTOR_DB_PERSIST_DIR", "./chroma_db")
    }
    
    config = {
        "llm": llm_config,
        "embedding": embedding_config,
        "reranker": reranker_config,
        "vector_store": vector_store_config
    }
    
    # 移除空的配置
    return {k: v for k, v in config.items() if v}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件
    
    优先从环境变量加载配置，如果没有设置环境变量则尝试从 YAML 文件加载
    
    【参数】
        config_path: 配置文件路径，默认 config/default.yaml
        如果传入了路径且文件存在，则优先使用 YAML 配置
        否则使用环境变量配置
    
    【返回】
        配置字典，环境变量已展开
    """
    # 加载 .env 文件
    _load_env_file()
    
    # 首先尝试从环境变量加载
    env_config = _load_from_env()
    
    # 如果环境变量有配置，直接返回
    if env_config.get("llm", {}).get("provider"):
        return env_config
    
    # 如果没有环境变量配置，尝试从 YAML 加载
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            config = _expand_env_vars(config)
            return config
    
    # 返回环境变量配置（可能为空）
    return env_config


class Config:
    """配置类 - 提供属性方式访问配置"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的路径"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config