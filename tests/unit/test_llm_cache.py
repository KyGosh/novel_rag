import contextlib

from pydantic import BaseModel

from novel_rag.llm.cache import CachedChatModel, cache_key
from novel_rag.llm.fake import FakeChatModel
from novel_rag.llm.usage import UsageTracker


class Person(BaseModel):
    name: str
    age: int


MSGS = [{"role": "user", "content": "你好"}]


def _cached(responder=None):
    inner = FakeChatModel(responder)
    tracker = UsageTracker()
    return CachedChatModel(inner, {}, tracker), inner, tracker


def test_second_identical_call_hits_cache():
    model, inner, tracker = _cached()
    r1 = model.chat(MSGS)
    r2 = model.chat(MSGS)
    assert inner.calls == 1
    assert not r1.cached and r2.cached
    assert r1.text == r2.text
    assert tracker.summary()["fake/fake-chat"]["cache_hits"] == 1


def test_prompt_version_and_messages_change_key():
    model, inner, _ = _cached()
    model.chat(MSGS, prompt_version="v1")
    model.chat(MSGS, prompt_version="v2")
    model.chat([{"role": "user", "content": "再见"}], prompt_version="v1")
    assert inner.calls == 3


def test_key_depends_on_schema_and_identity():
    k = cache_key({"model": "a"}, MSGS, None, None)
    assert k != cache_key({"model": "b"}, MSGS, None, None)
    assert k != cache_key({"model": "a"}, MSGS, Person, None)
    assert k == cache_key({"model": "a"}, list(MSGS), None, None)


def test_structured_result_restored_from_cache():
    model, inner, _ = _cached(lambda m: '{"name": "甲", "age": 3}')
    r1 = model.chat(MSGS, schema=Person)
    r2 = model.chat(MSGS, schema=Person)
    assert inner.calls == 1
    assert isinstance(r2.parsed, Person) and r2.parsed == r1.parsed


async def test_async_path_shares_cache():
    model, inner, _ = _cached()
    model.chat(MSGS)
    r = await model.achat(MSGS)
    assert r.cached and inner.calls == 1


def test_failed_call_is_not_cached():
    store: dict = {}
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    model = CachedChatModel(FakeChatModel(flaky), store, UsageTracker())
    with contextlib.suppress(RuntimeError):
        model.chat(MSGS)
    assert store == {}
    assert model.chat(MSGS).text == "ok"
