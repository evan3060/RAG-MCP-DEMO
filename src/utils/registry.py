"""
插件注册中心 - 管理可插拔组件

【设计模式】注册表模式 (Registry Pattern)
允许运行时动态注册和获取组件实现
"""

from typing import Any, Callable, Dict, Type


class Registry:
    """组件注册表"""

    _registry: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, component_type: str, name: str):
        """注册装饰器

        【参数】
        component_type: 组件类型 (llm, embedder, vector_store, etc.)
        name: 组件名称标识
        """
        def decorator(wrapped_class: Type):
            if component_type not in cls._registry:
                cls._registry[component_type] = {}

            cls._registry[component_type][name] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create(cls, component_type: str, name: str, config: Dict[str, Any]) -> Any:
        """创建组件实例

        【参数】
        component_type: 组件类型
        name: 组件名称
        config: 配置字典，传递给组件构造函数
        """
        if component_type not in cls._registry:
            raise ValueError(f"未知的组件类型: {component_type}")

        if name not in cls._registry[component_type]:
            available = list(cls._registry[component_type].keys())
            raise ValueError(
                f"未知的 {component_type} 组件: {name}. "
                f"可用选项: {available}"
            )

        component_class = cls._registry[component_type][name]
        return component_class(config)

    @classmethod
    def list_components(cls, component_type: str) -> list:
        """列出某类型的所有可用组件"""
        return list(cls._registry.get(component_type, {}).keys())
