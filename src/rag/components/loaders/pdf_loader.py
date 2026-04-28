"""PDF 文档加载器"""

import re
from pathlib import Path
from typing import List

from llama_index.core.schema import Document

from src.rag.components.loaders.base import BaseLoader


def clean_text(text: str) -> str:
    """清理文本，移除控制字符和PDF提取产生的乱码"""
    if not text:
        return text
    # 移除控制字符 (C0: \x00-\x1F 除换行/回车/制表符, C1: \x80-\x9F)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    # 移除Unicode特殊控制字符
    text = ''.join(char for char in text if not (0x00 <= ord(char) <= 0x1F or 0x7F <= ord(char) <= 0x9F))
    # 清理看起来像乱码的短序列
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x80-\x9F]{2,}', '', text)
    return text


class PDFLoader(BaseLoader):
    """PDF 文档加载器 - 支持解析 PDF 文本内容"""

    def load_data(self, file_path: Path, extra_info: dict = None) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("PDF 解析需要 pypdf 库。请安装: pip install pypdf")

        reader = PdfReader(str(file_path))
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text or not raw_text.strip():
                continue

            # 清理提取的文本
            cleaned_text = clean_text(raw_text)

            documents.append(Document(
                text=cleaned_text,
                metadata={
                    'source': str(file_path),
                    'type': 'pdf',
                    'page_number': page_num,
                    'total_pages': len(reader.pages),
                    'file_name': file_path.name,
                }
            ))

        if not documents:
            documents.append(Document(
                text=f"[PDF '{file_path.name}' 未能提取文本，可能是扫描件]",
                metadata={'source': str(file_path), 'type': 'pdf', 'extracted': False}
            ))

        return documents
