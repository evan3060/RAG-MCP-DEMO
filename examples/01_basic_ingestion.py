"""
示例 1: 基础文档上传

【运行方式】
python examples/01_basic_ingestion.py /path/to/documents
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.llamaindex.pipeline import RAGPipeline
from src.utils.config import load_config


async def main(documents_path: str):
    """上传文档并构建索引"""
    print(f"📚 开始上传文档: {documents_path}")

    # 加载配置
    config = load_config()

    # 初始化 Pipeline
    pipeline = RAGPipeline(config)

    # 构建索引
    await pipeline.build_index(documents_path)

    print("✅ 文档上传完成！")
    print(f"   索引已保存到: ./data/chroma_db/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python 01_basic_ingestion.py <documents_path>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
