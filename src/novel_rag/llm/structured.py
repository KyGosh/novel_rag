"""结构化输出的通用部分：构造 response_format、从模型输出中解析并校验 pydantic 对象。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from novel_rag.llm.base import Message

StructuredMode = Literal["json_schema", "json_object"]

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


def response_format_for(schema: type[BaseModel], mode: StructuredMode) -> dict[str, Any]:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    return {"type": "json_object"}


def schema_instruction(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return f"只输出一个 JSON 对象，不要输出其他内容。JSON 必须符合以下 JSON Schema：\n{schema_json}"


def with_schema_instruction(messages: Sequence[Message], schema: type[BaseModel]) -> list[Message]:
    """json_object 模式下服务端不知道 schema，需要写进 system prompt。"""
    msgs = [dict(m) for m in messages]
    instruction = schema_instruction(schema)
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{instruction}"
    else:
        msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """去掉可能的 ```json 围栏后校验。失败抛 ValidationError。"""
    m = _FENCE.match(text)
    payload = m.group(1) if m else text.strip()
    return schema.model_validate_json(payload)


def validation_feedback(text: str, err: ValidationError) -> list[Message]:
    """校验失败后追加到对话中的纠错消息。"""
    return [
        {"role": "assistant", "content": text},
        {
            "role": "user",
            "content": f"上面的输出没有通过校验：{err.errors(include_url=False)}。请修正后只输出合法的 JSON。",
        },
    ]
