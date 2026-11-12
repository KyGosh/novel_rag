"""P4：llm/ 的 function calling（同步 / 异步）、非法参数原样上交、缓存覆盖工具调用与采样编号、脚本化 Fake 模型。"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from novel_rag.llm.base import ToolCall, tool_call_message, tool_result_message
from novel_rag.llm.cache import CachedChatModel, cache_key
from novel_rag.llm.fake import Call, FakeChatModel, Say, ScriptedChatModel
from novel_rag.llm.usage import UsageTracker
from tests.unit.test_openai_chat import _model

TOOLS = [
    {
        "name": "search_text",
        "description": "检索原文",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
]
MSGS = [{"role": "user", "content": "谁是北境之王？"}]


def _tool_resp(calls, content=None):
    tcs = [
        SimpleNamespace(id=f"c{i}", type="function", function=SimpleNamespace(name=n, arguments=a))
        for i, (n, a) in enumerate(calls)
    ]
    msg = SimpleNamespace(content=content, reasoning_content=None, tool_calls=tcs)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=8),
        model="fake-model",
    )


def test_tools_are_sent_in_openai_format_and_calls_parsed():
    m, comps, tracker = _model([_tool_resp([("search_text", '{"query": "北境"}')])])
    r = m.chat(MSGS, tools=TOOLS)
    req = comps.requests[0]
    assert req["tools"] == [
        {"type": "function", "function": {"name": "search_text", "description": "检索原文",
                                          "parameters": TOOLS[0]["parameters"]}}
    ]  # fmt: skip
    assert r.text == ""
    assert r.tool_calls == (ToolCall("c0", "search_text", '{"query": "北境"}', {"query": "北境"}, None),)
    assert tracker.summary()["t/fake-model"]["prompt_tokens"] == 30


def test_no_tools_param_when_not_given():
    m, comps, _ = _model([_tool_resp([], content="好")])
    r = m.chat(MSGS)
    assert "tools" not in comps.requests[0]
    assert r.text == "好" and r.tool_calls == ()


def test_invalid_arguments_are_passed_up_not_raised():
    m, _, _ = _model([_tool_resp([("search_text", '{"query": 北境'), ("search_text", "[1, 2]")])])
    r = m.chat(MSGS, tools=TOOLS)
    bad_json, not_obj = r.tool_calls
    assert bad_json.args is None and "JSON" in bad_json.args_error and bad_json.arguments == '{"query": 北境'
    assert not_obj.args is None and "对象" in not_obj.args_error


def test_async_tool_calls():
    m, _, _ = _model([_tool_resp([("search_text", '{"query": "x"}')])], is_async=True)
    r = asyncio.run(m.achat(MSGS, tools=TOOLS))
    assert r.tool_calls[0].args == {"query": "x"}


def test_schema_not_parsed_when_model_calls_tool():
    class A(BaseModel):
        x: int

    m, _, _ = _model([_tool_resp([("search_text", "{}")])])
    r = m.chat(MSGS, schema=A, tools=TOOLS)
    assert r.parsed is None and r.tool_calls[0].args == {}


def test_tool_messages_roundtrip_format():
    calls = (ToolCall.parse("c1", "search_text", '{"query": "a"}'),)
    msg = tool_call_message("", calls)
    assert msg == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_text", "arguments": '{"query": "a"}'}}
        ],
    }
    assert tool_result_message("c1", "结果") == {"role": "tool", "tool_call_id": "c1", "content": "结果"}


# ---------------------------------------------------------------- 缓存


def test_cache_key_backward_compatible_without_tools_and_sample():
    ident = {"model": "m"}
    assert cache_key(ident, MSGS, None, "p.v1") == cache_key(ident, MSGS, None, "p.v1", None, 0)
    assert cache_key(ident, MSGS, None, "p.v1", TOOLS) != cache_key(ident, MSGS, None, "p.v1")
    assert cache_key(ident, MSGS, None, "p.v1", sample=1) != cache_key(ident, MSGS, None, "p.v1")
    other_tools = [{**TOOLS[0], "description": "检索原文（改）"}]
    assert cache_key(ident, MSGS, None, "p.v1", TOOLS) != cache_key(ident, MSGS, None, "p.v1", other_tools)


def test_cached_model_stores_and_replays_tool_calls():
    inner = ScriptedChatModel([Call.one("search_text", {"query": "北境"}), Call.one("search_text", {"query": "南方"})])
    model = CachedChatModel(inner, {}, UsageTracker())
    r1 = model.chat(MSGS, tools=TOOLS, prompt_version="agent.v1")
    r2 = model.chat(MSGS, tools=TOOLS, prompt_version="agent.v1")
    assert r2.cached and r2.tool_calls == r1.tool_calls and r1.tool_calls[0].args == {"query": "北境"}
    # 换采样编号：独立样本，不命中缓存
    r3 = model.chat(MSGS, tools=TOOLS, prompt_version="agent.v1", sample=1)
    assert not r3.cached and r3.tool_calls[0].args == {"query": "南方"}


def test_cached_model_history_with_tool_messages_changes_key():
    inner = FakeChatModel(lambda msgs: "ok")
    model = CachedChatModel(inner, {}, UsageTracker())
    calls = (ToolCall.parse("c1", "search_text", '{"query": "a"}'),)
    h1 = [*MSGS, tool_call_message("", calls), tool_result_message("c1", "结果一")]
    h2 = [*MSGS, tool_call_message("", calls), tool_result_message("c1", "结果二")]
    model.chat(h1)
    model.chat(h2)
    model.chat(h1)
    assert inner.calls == 2


# ---------------------------------------------------------------- 脚本化 Fake


def test_scripted_model_by_prompt_version_and_callable_steps():
    class Out(BaseModel):
        answer: str

    m = ScriptedChatModel(
        {
            "agent.v1": [Call((("a", {"x": 1}), ("b", '{"bad"'))), lambda msgs, tools: Say(f"tools={len(tools)}")],
            "answer.v1": [Say('{"answer": "琼恩"}')],
        }
    )
    r = m.chat(MSGS, tools=TOOLS, prompt_version="agent.v1")
    assert [c.name for c in r.tool_calls] == ["a", "b"] and r.tool_calls[1].args is None
    assert m.chat(MSGS, tools=TOOLS, prompt_version="agent.v1").text == "tools=1"
    assert m.chat(MSGS, schema=Out, prompt_version="answer.v1").parsed == Out(answer="琼恩")
    assert m.history[0]["tools"] == ["search_text"]
    with pytest.raises(AssertionError, match="脚本已用完"):
        m.chat(MSGS, prompt_version="agent.v1")
