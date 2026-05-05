"""
OpenAI 兼容接口 LLM 实现（使用自定义HTTP客户端）

支持：千帆 Coding 等 OpenAI 兼容 API
"""

from typing import List, Optional
import json

import httpx

from src.rag.components.llms.base import BaseLLM, LLMMessage, LLMResponse
from src.utils.registry import Registry


@Registry.register("llm", "openai")
@Registry.register("llm", "siliconflow")
@Registry.register("llm", "myapi")
class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI 兼容接口 LLM 实现（使用自定义HTTP客户端）
    
    处理非标准API响应格式
    """

    def __init__(self, config: dict):
        """初始化 OpenAI 兼容 LLM"""
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.default_temperature = config.get("temperature", 0.7)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")

        super().__init__(config)

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

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": openai_messages,
                    "temperature": temperature or self.default_temperature,
                    **kwargs
                }
            )

            response.raise_for_status()
            
            # 处理响应前面的空白字符
            text = response.text.strip()
            data = json.loads(text)

            # 提取回答内容
            content = data["choices"][0]["message"]["content"]
            
            # 提取使用量信息（如果有）
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model_name
            )
