"""
千帆 LLM 实现 - 百度智能云大模型 API

【学习要点】
1. 如何实现基类接口
2. 如何处理 API 调用
3. 如何转换数据格式
"""

from typing import List, Optional

import qianfan

from src.rag.components.llms.base import BaseLLM, LLMMessage, LLMResponse


class QianfanLLM(BaseLLM):
    """
    千帆 LLM 实现

    【支持的模型】
    - ERNIE-Bot-4: 百度最新旗舰模型
    - ERNIE-Bot: 通用对话模型
    - ERNIE-Bot-turbo: 轻量快速模型

    【使用示例】
    ```python
    llm = QianfanLLM({
        "api_key": "your-key",
        "secret_key": "your-secret",
        "model": "ERNIE-Bot-4",
        "temperature": 0.7
    })

    response = await llm.generate([
        LLMMessage(role="user", content="你好")
    ])
    ```
    """

    def __init__(self, config: dict):
        """初始化千帆 LLM"""
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.model_name = config.get("model", "ERNIE-Bot-4")
        self.default_temperature = config.get("temperature", 0.7)

        super().__init__(config)

        self.client = qianfan.ChatCompletion(
            ak=self.api_key,
            sk=self.secret_key
        )

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "千帆 LLM 需要提供 api_key 和 secret_key"
            )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        调用千帆 API 生成回答

        【参数转换】
        我们的 LLMMessage -> 千帆的 message 格式
        """
        qianfan_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = self.client.do(
            model=self.model_name,
            messages=qianfan_messages,
            temperature=temperature or self.default_temperature,
            **kwargs
        )

        result = response.body

        return LLMResponse(
            content=result.get("result", ""),
            usage=result.get("usage", {}),
            model=self.model_name
        )
