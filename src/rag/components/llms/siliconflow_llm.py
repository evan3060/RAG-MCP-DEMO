"""
SiliconFlow LLM 实现 - OpenAI 兼容格式

【学习要点】
1. SiliconFlow 提供 OpenAI 兼容的 API
2. 使用 httpx 进行异步 HTTP 请求
3. 统一的 LLMMessage/LLMResponse 接口
"""

from typing import List, Optional

import httpx

from src.rag.components.llms.base import BaseLLM, LLMMessage, LLMResponse


class SiliconFlowLLM(BaseLLM):
    """
    SiliconFlow LLM 实现

    【支持的模型】
    - deepseek-ai/DeepSeek-V3
    - deepseek-ai/DeepSeek-R1
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    - Qwen/Qwen2.5-72B-Instruct
    - 等更多模型: https://siliconflow.cn/models

    【使用示例】
    ```python
    llm = SiliconFlowLLM({
        "api_key": "sk-your-key",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "temperature": 0.7,
        "base_url": "https://api.siliconflow.cn/v1"
    })

    response = await llm.generate([
        LLMMessage(role="user", content="你好")
    ])
    ```
    """

    API_BASE = "https://api.siliconflow.cn/v1"

    def __init__(self, config: dict):
        """初始化 SiliconFlow LLM"""
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", "deepseek-ai/DeepSeek-V3")
        self.default_temperature = config.get("temperature", 0.7)
        self.base_url = config.get("base_url", self.API_BASE)

        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValueError(
                "SiliconFlow LLM 需要提供 api_key"
            )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        调用 SiliconFlow API 生成回答

        【API 文档】
        https://docs.siliconflow.cn/api-reference/chat-completions/chat-completions
        """
        # 转换消息格式
        siliconflow_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": siliconflow_messages,
                    "temperature": temperature or self.default_temperature,
                    **kwargs
                },
                timeout=60.0
            )

            response.raise_for_status()
            data = response.json()

            # 解析响应
            choice = data["choices"][0]
            message = choice["message"]

            return LLMResponse(
                content=message.get("content", ""),
                usage=data.get("usage", {}),
                model=self.model_name
            )
