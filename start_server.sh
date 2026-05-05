#!/bin/bash
cd /home/evan/myproject/RAG-MCP-DEMO
source venv/bin/activate
exec uvicorn src.api.server:app --host 192.168.0.15 --port 8000
