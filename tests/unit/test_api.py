"""P6 本地 Web 演示的接口冒烟测试：用假模型 + 假后端组装的问答服务，检查 SSE 事件的顺序与内容。不连真实服务。"""

import json

from fastapi.testclient import TestClient

from novel_rag.api.app import create_app
from tests.unit.test_service import service
from tests.unit.tool_fakes import make_backends


def events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_index_and_meta():
    c = TestClient(create_app(service))
    r = c.get("/")
    assert r.status_code == 200 and "<title>" in r.text
    m = c.get("/api/meta").json()
    assert [x["id"] for x in m["modes"]] == ["rag", "agent"]
    assert m["volumes"][0]["book_id"] == "demo-1" and m["volumes"][0]["chapters"]


def test_ask_streams_steps_then_result():
    svc = service()
    c = TestClient(create_app(lambda: svc))
    r = c.get("/api/ask", params={"q": "白狐后来被取了什么名字？", "mode": "agent", "volume": "demo-1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r.text)
    kinds = [k for k, _ in ev]
    assert kinds[-1] == "result" and set(kinds[:-1]) == {"step"}
    steps = [d["type"] for k, d in ev if k == "step"]
    assert steps[:2] == ["start", "plan"] and steps[-1] == "answer"
    assert steps.index("act") < steps.index("tool") < steps.index("reflect")
    res = ev[-1][1]
    assert res["answer"] == "雪团" and res["citations"][0]["where"] == "北岭记 · 柳月"
    assert res["max_reading_order"] == svc.reading_order_of("demo-1")
    for c_ in make_backends().chunks.chunks:  # 整个事件流都不含整段原文
        assert c_["text"] not in r.text


def test_ask_reports_bad_input_as_error_event():
    c = TestClient(create_app(service))
    ev = events(c.get("/api/ask", params={"q": "问题", "mode": "closed"}).text)
    assert ev[-1][0] == "error" and "未知模式" in ev[-1][1]["message"]
    ev = events(c.get("/api/ask", params={"q": "问题", "volume": "demo-9"}).text)
    assert ev[-1][0] == "error"
    assert c.get("/api/ask", params={"q": "问" * 201}).status_code == 422
