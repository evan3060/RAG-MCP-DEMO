"""
嵌入模型基类 - 将文本转换为向量

【什么是嵌入?】
把文字转换成数字向量，让计算机能"理解"语义
例如: "狗" -> [0.1, -0.5, 0.8, ...] (1024维向量)

【为什么需要?】
- 计算机只能处理数字，不能直接理解文字
- 相似语义的文本有相似的向量（余弦相似度高）
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """嵌入模型抽象基类"""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "")
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        嵌入单条文本

        【参数】
        text: 输入文本

        【返回】
        向量列表，如 [0.1, -0.5, 0.8, ...]
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入多条文本（更高效）

        【参数】
        texts: 文本列表

        【返回】
        向量列表的列表
        """
        pass
