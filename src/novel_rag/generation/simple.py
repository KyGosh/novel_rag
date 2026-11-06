"""最简生成：把检索到的前若干个 chunk（附卷名、章节标题、chunk_id）交给生成模型作答。P1 只要求能回答，不做引用校验。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from novel_rag.llm.base import ChatModel, Message
from novel_rag.prompts import Prompt, load_prompt
from novel_rag.retrieval.base import Hit


def _sections(prompt: Prompt) -> dict[str, str]:
    """prompt 文件按 `## system` / `## user` 分节。"""
    parts: dict[str, str] = {}
    current = None
    for line in prompt.text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            parts[current] = ""
        elif current is not None:
            parts[current] += line + "\n"
    return {k: v.strip() for k, v in parts.items()}


def build_messages(question: str, hits: Sequence[Hit], prompt: Prompt, book_titles: dict[str, str]) -> list[Message]:
    blocks = []
    for i, h in enumerate(hits, 1):
        f = h.fields
        where = " ".join(x for x in (book_titles.get(f.get("book_id", ""), ""), f.get("chapter_title", "")) if x)
        blocks.append(f"[{i}] 《{where}》 {h.chunk_id}\n{f.get('text', '')}")
    sec = _sections(prompt)
    user = sec["user"].replace("{question}", question).replace("{context}", "\n\n".join(blocks))
    return [{"role": "system", "content": sec["system"]}, {"role": "user", "content": user}]


def answer(
    question: str, hits: Sequence[Hit], chat: ChatModel, *, book_titles: dict[str, str], prompt: Prompt | None = None
) -> dict[str, Any]:
    prompt = prompt or load_prompt("answer")
    result = chat.chat(build_messages(question, hits, prompt, book_titles), prompt_version=prompt.version_id)
    return {
        "answer": result.text,
        "chunk_ids": [h.chunk_id for h in hits],
        "model": result.model,
        "prompt_version": prompt.version_id,
        "cached": result.cached,
        "usage": {"prompt_tokens": result.usage.prompt_tokens, "completion_tokens": result.usage.completion_tokens},
    }
