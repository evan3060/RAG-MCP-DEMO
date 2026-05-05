"""
OpenAI 兼容接口 LLM 实现

支持：千帆 Coding 等 OpenAI 兼容 API
"""

from typing import List, Optional

from openai import AsyncOpenAI

from src.rag.components.llms.base import BaseLLM, LLMMessage, LLMResponse
from src.utils.registry import Registry


@Registry.register("llm", "openai")
@Registry.register("llm", "siliconflow")
@Registry.register("llm", "myapi")
class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI 兼容接口 LLM 实现

    【支持的模型】
    - 千帆 Coding: kimi-k2.5, ernie-bot 等
    - 任何 OpenAI 兼容接口

    【使用示例】
    ```python
    llm = OpenAICompatibleLLM({
        "api_key": "your-key",
        "model": "kimi-k2.5",
        "base_url": "https://qianfan.baidubce.com/v2/coding",
        "temperature": 0.7
    })

    response = await llm.generate([
        LLMMessage(role="user", content="你好")
    ])
    ```
    """

    def __init__(self, config: dict):
        """初始化 OpenAI 兼容 LLM"""
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.default_temperature = config.get("temperature", 0.7)
        self.base_url = config.get("base_url")  # OpenAI 兼容接口的 base_url

        super().__init__(config)

        # 初始化 OpenAI 客户端
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValueError("OpenAI 兼容 LLM 需要提供 api_key")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        调用 OpenAI 兼容 API 生成回答
        """
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=temperature or self.default_temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            model=self.model_name
        )
