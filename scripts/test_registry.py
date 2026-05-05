#!/usr/bin/env python3
"""测试组件工厂和注册表"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.components.factory import list_available_components
from src.rag.components import *

print("=" * 60)
print("组件注册表测试")
print("=" * 60)

components = list_available_components()

print("\n可用的 LLM 提供商:")
for provider in components['llm']:
    print(f"  - {provider}")

print("\n可用的 Embedding 提供商:")
for provider in components['embedder']:
    print(f"  - {provider}")

print("\n可用的 Reranker 提供商:")
for provider in components['reranker']:
    print(f"  - {provider}")

print("\n可用的 Vector Store 提供商:")
for provider in components['vector_store']:
    print(f"  - {provider}")

print("\n" + "=" * 60)
print("✅ 组件注册成功！")
print("=" * 60)
