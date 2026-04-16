"""
配置管理模块 - 加载和管理应用配置

【设计说明】
- 支持 YAML 配置文件
- 自动展开环境变量（如 ${API_KEY}）
- 提供类型安全的配置访问
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


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


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """加载配置文件

    【参数】
    config_path: 配置文件路径，默认 config/default.yaml

    【返回】
    配置字典，环境变量已展开
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = _expand_env_vars(config)
    return config


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
