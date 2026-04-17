#!/usr/bin/env python3
"""
启动 API 服务器脚本

使用方法:
    python scripts/start_api_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from src.api.server import app

if __name__ == "__main__":
    print("🚀 启动 RAG API 服务器...")
    print("📚 API 文档: http://localhost:8000/docs")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
