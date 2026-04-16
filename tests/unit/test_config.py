"""测试配置模块"""

import pytest
import os
import tempfile
from pathlib import Path

from src.utils.config import load_config, Config, _expand_env_vars


def test_expand_env_vars():
    """测试环境变量展开"""
    os.environ['TEST_KEY'] = 'test_value'

    result = _expand_env_vars('${TEST_KEY}')
    assert result == 'test_value'

    result = _expand_env_vars('${UNDEFINED:-default}')
    assert result == 'default'

    result = _expand_env_vars({'key': '${TEST_KEY}'})
    assert result == {'key': 'test_value'}

    result = _expand_env_vars(['${TEST_KEY}'])
    assert result == ['test_value']


def test_load_config_from_file():
    """测试从文件加载配置"""
    config_content = """
server:
  mode: stdio
  port: 8000
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config['server']['mode'] == 'stdio'
        assert config['server']['port'] == 8000
    finally:
        os.unlink(temp_path)


def test_config_class():
    """测试 Config 类"""
    config_dict = {
        'server': {'mode': 'stdio', 'port': 8000}
    }
    config = Config(config_dict)

    assert config.get('server.mode') == 'stdio'
    assert config.get('nonexistent', 'default') == 'default'
    assert 'server' in config
