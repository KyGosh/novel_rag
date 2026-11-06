"""集成测试：需要先 `docker compose up -d`、本地 GPU 和可用的 LLM 接口（默认为本地 DeepSeek 代理）。运行：uv run pytest -m integration"""

import pytest

from scripts.check_env import CHECKS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", list(CHECKS))
def test_environment(name):
    summary, _ = CHECKS[name]()
    print(summary)
