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
        """初始化千帆 LLM

        【认证方式】
        1. API Key 方式（推荐，Coding Plan 使用）: 只提供 api_key
        2. AK/SK 方式（标准方式）: 提供 api_key + secret_key
        """
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.model_name = config.get("model", "ERNIE-Bot-4")
        self.default_temperature = config.get("temperature", 0.7)
        self.base_url = config.get("base_url")  # 支持自定义 base_url

        super().__init__(config)

        # 如果配置了自定义 base_url，通过环境变量设置
        # 千帆 SDK 使用 QIANFAN_BASE_URL 环境变量
        if self.base_url:
            import os
            os.environ["QIANFAN_BASE_URL"] = self.base_url

        # 根据认证方式初始化客户端
        if self.secret_key:
            # AK/SK 方式
            self.client = qianfan.ChatCompletion(
                ak=self.api_key,
                sk=self.secret_key
            )
        else:
            # API Key 方式（Coding Plan 代理常用）
            self.client = qianfan.ChatCompletion(
                access_token=self.api_key
            )

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValueError(
                "千帆 LLM 需要提供 api_key "
                "（或同时提供 api_key 和 secret_key）"
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
