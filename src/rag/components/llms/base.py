"""
LLM 基类设计 - 面向初级开发者的接口抽象教学

【学习要点】
1. 什么是抽象基类 (ABC)?
   - 定义接口规范，强制子类实现特定方法
   - 保证不同 LLM 实现有一致的使用方式

2. 为什么要使用基类?
   - 解耦: 上层代码不依赖具体 LLM 实现
   - 可扩展: 新增 LLM 只需实现基类接口
   - 可测试: 易于 Mock 和单元测试
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """
    消息模型 - 统一不同 LLM 的消息格式

    【类比理解】
    就像写信，每封信都有：
    - role: 是谁写的（用户/助手/系统）
    - content: 写了什么内容
    """
    role: str       # "system", "user", "assistant"
    content: str    # 消息内容


class LLMResponse(BaseModel):
    """
    LLM 响应模型

    【字段说明】
    - content: AI 生成的回答文本
    - usage: Token 使用量（用于计费和分析）
    - model: 使用的模型名称
    """
    content: str
    usage: dict = {}
    model: str = ""


class BaseLLM(ABC):
    """
    LLM 抽象基类 - 所有 LLM 实现的接口规范

    【设计模式】模板方法模式 (Template Method)
    基类定义算法骨架，子类实现具体步骤
    """

    def __init__(self, config: dict):
        """
        初始化方法

        【参数】
        config: 配置字典，包含 api_key, model, temperature 等
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        验证配置 - 子类必须实现

        【为什么需要】
        尽早发现配置错误，避免运行时出错
        """
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本 - 核心方法

        【参数】
        messages: 对话历史，格式 [{"role": "user", "content": "你好"}]
        temperature: 创造性参数 (0-2)，越高回答越随机
        **kwargs: 额外的模型特定参数

        【返回】
        LLMResponse 对象，包含生成的文本和元信息
        """
        pass

    async def stream_generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成 - 逐字返回（用于打字机效果）

        【使用场景】
        前端需要实时显示 AI 回答，而不是等全部生成完
        """
        response = await self.generate(messages, **kwargs)
        for char in response.content:
            yield char
