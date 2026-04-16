"""文档加载器基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llama_index.core.schema import Document


class BaseLoader(ABC):
    """文档加载器抽象基类"""

    @abstractmethod
    def load_data(self, file_path: Path) -> List[Document]:
        """加载文件并返回文档列表"""
        pass
