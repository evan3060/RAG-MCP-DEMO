"""
LlamaIndex LLM 适配器

将我们的 BaseLLM 适配到 LlamaIndex 的 LLM 接口
"""

from typing import Any, List, Optional, Sequence

from llama_index.core.llms import (
    LLM as LlamaIndexLLM,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.base.llms.types import MessageRole
from pydantic import PrivateAttr

from src.rag.components.llms.base import BaseLLM, LLMMessage


class LlamaIndexLLMAdapter(LlamaIndexLLM):
    """
    适配器：将 BaseLLM 包装为 LlamaIndex 兼容的 LLM
    """

    _llm: BaseLLM = PrivateAttr(default=None)

    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def metadata(self):
        """返回模型元数据"""
        from llama_index.core.base.llms.types import LLMMetadata
        return LLMMetadata(
            context_window=4096,
            num_output=2048,
            model_name=getattr(self._llm, 'model_name', 'unknown'),
        )

    def _convert_messages(self, messages: Sequence[ChatMessage]) -> List[LLMMessage]:
        """转换 LlamaIndex 消息格式为我们的格式"""
        return [
            LLMMessage(role=msg.role.value, content=msg.content)
            for msg in messages
        ]

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """同步聊天接口"""
        import asyncio
        our_messages = self._convert_messages(messages)
        response = asyncio.run(self._llm.generate(our_messages))

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content
            ),
            raw=response.dict()
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """同步完成接口"""
        import asyncio
        messages = [LLMMessage(role="user", content=prompt)]
        response = asyncio.run(self._llm.generate(messages))

        return CompletionResponse(
            text=response.content,
            raw=response.dict()
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """异步聊天接口"""
        our_messages = self._convert_messages(messages)
        response = await self._llm.generate(our_messages)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content
            ),
            raw=response.dict()
        )

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """异步完成接口"""
        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.generate(messages)

        return CompletionResponse(
            text=response.content,
            raw=response.dict()
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """流式聊天（暂未实现）"""
        raise NotImplementedError("流式聊天暂未实现")

    def stream_complete(self, prompt: str, **kwargs: Any):
        """流式完成（暂未实现）"""
        raise NotImplementedError("流式完成暂未实现")

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """异步流式聊天（暂未实现）"""
        raise NotImplementedError("异步流式聊天暂未实现")

    async def astream_complete(self, prompt: str, **kwargs: Any):
        """异步流式完成（暂未实现）"""
        raise NotImplementedError("异步流式完成暂未实现")
