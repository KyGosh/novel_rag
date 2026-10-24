"""用假的 OpenAI 客户端测试重试、结构化输出、思考内容丢弃和用量记录，不访问网络。"""

from types import SimpleNamespace

import httpx2 as httpx  # openai 3.x 的 HTTP 层
import openai
import pytest
from pydantic import BaseModel
from tenacity import wait_none

from novel_rag.config import LLMConfig, LLMTaskConfig
from novel_rag.llm.base import StructuredOutputError
from novel_rag.llm.openai_chat import ConcurrencyLimiter, OpenAICompatChatModel, is_retryable
from novel_rag.llm.usage import UsageTracker


class Answer(BaseModel):
    city: str
    count: int


def _resp(content, prompt=10, completion=5, reasoning=None):
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        model="fake-model",
    )


class ScriptedCompletions:
    """按顺序返回预设结果；元素是异常时抛出。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class AsyncScriptedCompletions(ScriptedCompletions):
    async def create(self, **kwargs):  # type: ignore[override]
        return super().create(**kwargs)


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _model(script, *, structured_mode="json_schema", is_async=False, **task_kwargs):
    comps = (AsyncScriptedCompletions if is_async else ScriptedCompletions)(script)
    llm_cfg = LLMConfig(max_retries=2, structured_mode=structured_mode, structured_max_attempts=2)
    task_cfg = LLMTaskConfig(model="m", **task_kwargs)
    tracker = UsageTracker()
    m = OpenAICompatChatModel(
        "t", task_cfg, llm_cfg, tracker, ConcurrencyLimiter(2), client=_client(comps), async_client=_client(comps)
    )
    m.retry_wait = wait_none()
    return m, comps, tracker


def _conn_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x"))


def _status_error(cls, code):
    req = httpx.Request("POST", "http://x")
    return cls("err", response=httpx.Response(code, request=req), body=None)


def test_is_retryable():
    assert is_retryable(_status_error(openai.RateLimitError, 429))
    assert is_retryable(_status_error(openai.InternalServerError, 503))
    assert is_retryable(_conn_error())
    assert not is_retryable(_status_error(openai.BadRequestError, 400))
    assert not is_retryable(ValueError())


def test_plain_chat_drops_reasoning_and_records_usage():
    m, comps, tracker = _model(
        [_resp("最终答案", reasoning="思考过程……")], temperature=0.3, extra_body={"thinking": {"type": "disabled"}}
    )
    r = m.chat([{"role": "user", "content": "q"}])
    assert r.text == "最终答案"
    assert "思考" not in r.text
    assert r.usage.total_tokens == 15
    req = comps.requests[0]
    assert req["temperature"] == 0.3 and req["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "response_format" not in req
    [rec] = tracker.records
    assert rec.prompt_tokens == 10 and not rec.cached


def test_retries_transient_errors_then_succeeds():
    m, comps, _ = _model([_status_error(openai.RateLimitError, 429), _conn_error(), _resp("ok")])
    assert m.chat([{"role": "user", "content": "q"}]).text == "ok"
    assert len(comps.requests) == 3


def test_gives_up_after_max_retries():
    m, _, _ = _model([_conn_error()] * 3)
    with pytest.raises(openai.APIConnectionError):
        m.chat([{"role": "user", "content": "q"}])


def test_non_retryable_error_raises_immediately():
    m, comps, _ = _model([_status_error(openai.BadRequestError, 400), _resp("never")])
    with pytest.raises(openai.BadRequestError):
        m.chat([{"role": "user", "content": "q"}])
    assert len(comps.requests) == 1


def test_structured_json_schema_mode():
    m, comps, _ = _model([_resp('```json\n{"city": "临冬城", "count": 2}\n```')])
    r = m.chat([{"role": "user", "content": "q"}], schema=Answer)
    assert r.parsed == Answer(city="临冬城", count=2)
    rf = comps.requests[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "Answer"


def test_structured_validation_failure_retries_with_feedback():
    m, comps, tracker = _model([_resp('{"city": "x"}'), _resp('{"city": "x", "count": 1}')])
    r = m.chat([{"role": "user", "content": "q"}], schema=Answer)
    assert r.parsed.count == 1 and r.attempts == 2
    assert r.usage.prompt_tokens == 20  # 两次请求的用量累加
    second_msgs = comps.requests[1]["messages"]
    assert second_msgs[-2]["role"] == "assistant" and second_msgs[-1]["role"] == "user"
    assert tracker.records[0].attempts == 2


def test_structured_gives_up():
    m, _, _ = _model([_resp("not json"), _resp("still not")])
    with pytest.raises(StructuredOutputError):
        m.chat([{"role": "user", "content": "q"}], schema=Answer)


def test_json_object_mode_injects_schema_into_system_prompt():
    m, comps, _ = _model([_resp('{"city": "a", "count": 1}')], structured_mode="json_object")
    m.chat([{"role": "system", "content": "你是助手"}, {"role": "user", "content": "q"}], schema=Answer)
    req = comps.requests[0]
    assert req["response_format"] == {"type": "json_object"}
    assert req["messages"][0]["content"].startswith("你是助手") and "count" in req["messages"][0]["content"]


async def test_async_chat_with_retry_and_schema():
    m, comps, _ = _model([_conn_error(), _resp('{"city": "a", "count": 1}')], is_async=True)
    r = await m.achat([{"role": "user", "content": "q"}], schema=Answer)
    assert r.parsed.city == "a" and len(comps.requests) == 2
