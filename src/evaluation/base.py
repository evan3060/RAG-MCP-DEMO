"""评估器基类 - 评估 RAG 系统质量"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class EvalResult:
    """评估结果"""
    query: str
    answer: str
    contexts: List[str]
    metrics: Dict[str, float]


class BaseEvaluator(ABC):
    """评估器抽象基类"""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: str = None
    ) -> EvalResult:
        """评估单个查询"""
        pass
