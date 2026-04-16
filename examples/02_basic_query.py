"""
示例 2: 基础查询

【运行方式】
python examples/02_basic_query.py "什么是RAG?"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.llamaindex.pipeline import RAGPipeline
from src.utils.config import load_config


async def main(question: str):
    """执行问答查询"""
    print(f"🤔 问题: {question}\n")

    config = load_config()
    pipeline = RAGPipeline(config)

    # 需要先构建索引才能查询
    # 如果索引已存在，可以跳过构建步骤

    result = await pipeline.ask(question)

    print(f"💡 回答:\n{result['answer']}\n")
    print("📚 参考来源:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. [{source['score']:.2f}] {source['content'][:100]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python 02_basic_query.py '你的问题'")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
