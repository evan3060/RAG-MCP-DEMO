"""pytest 配置和共享 fixtures"""

import pytest


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "siliconflow_api_key": "test-key",
        "qianfan_api_key": "test-key",
        "qianfan_secret_key": "test-secret",
        "chunk_size": 512,
        "top_k": 10
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("QIANFAN_API_KEY", "test-key")
    monkeypatch.setenv("QIANFAN_SECRET_KEY", "test-secret")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
