"""P4 工具层集成测试（真实 ES / Milvus / Neo4j / GPU）：输出格式、长度上限，以及防剧透断言——
在多个阅读进度下调用每个工具，观察中出现的 chunk、证据和位置都不超过阅读进度。不打印原文。"""

import pytest

from novel_rag.config import get_settings
from novel_rag.tools import ToolContext, Toolbox, build_backends

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def toolbox():
    from novel_rag.qa.env import load_env

    s = get_settings()
    env = load_env(s)
    return Toolbox(build_backends(s, env.scope, env.chunks, env.titles)), env


@pytest.mark.parametrize("max_ro", [5, 74, 150, 230])
def test_every_tool_respects_reading_progress(toolbox, max_ro):
    tb, env = toolbox
    ctx = ToolContext(max_reading_order=max_ro, book_ids=env.scope)
    ent = tb.call("find_entity", {"name": "琼恩"}, ctx)
    assert ent.ok
    eid = ent.meta["candidates"][0]
    calls = [
        ("search_text", {"query": "琼恩 长城 守夜人 誓言", "top_k": 8}),
        ("search_text", {"query": "冰原狼", "entity_id": eid, "top_k": 8}),
        ("search_summaries", {"query": "守夜人 总司令 选举", "top_k": 8}),
        ("read_summary", {"chapter": "asoiaf-1:0003"}),
        ("read_text", {"chunk_id": "asoiaf-1:0003:0002", "before": 3, "after": 3}),
        *[
            ("graph_query", {"op": op, "entity": eid, "limit": 30})
            for op in ("profile", "neighbors", "timeline", "mentions")
        ],
    ]
    for name, args in calls:
        r = tb.call(name, args, ctx)  # 越界时 Toolbox 抛 SpoilerLeak
        assert r.text and len(r.text) <= tb.hard_max_chars
        for cid in r.chunk_ids:
            if cid in env.spans:
                assert env.chunks.by_id(cid)["reading_order"] <= max_ro, (name, cid)
        assert all(x <= max_ro for x in r.reading_orders), name
        if r.ok and r.shown_chunk_ids:
            assert f"【{r.shown_chunk_ids[0]}" in r.text and " · " in r.text  # chunk_id + 卷名 · 章节标题
