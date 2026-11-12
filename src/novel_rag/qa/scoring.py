"""答案评分：规则匹配 + LLM judge。

1. 弃答（"原文未提及" / 空答案）直接记为 abstain，不送 judge。
2. 规则匹配：答案与标准答案、可接受的其他写法做规范化（统一间隔号、去标点和空白、小写）后的包含匹配，
   匹配上就算 correct。
3. 规则没匹配上的交给 judge（单独的任务配置），判定 correct / partial / wrong 并给出理由。judge 不看证据。
主指标是 correct 的比例，partial 另列，弃答率单独报告。judge 与人工的一致率见 scripts/judge_agreement.py。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel

from novel_rag.llm.base import ChatModel, StructuredOutputError
from novel_rag.prompts import Prompt, load_prompt

Verdict = Literal["correct", "partial", "wrong"]
LABELS = ("correct", "partial", "wrong", "abstain")

_MIDDOTS = "·•・‧∙⋅"
_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(s: str) -> str:
    """统一间隔号（全部去掉）、去标点和空白、全角转半角、小写。不做繁简或同义改写。"""
    s = unicodedata.normalize("NFKC", s)  # 只用于比较，不改动任何存储的文本
    for ch in _MIDDOTS:
        s = s.replace(ch, "")
    return _STRIP.sub("", s).lower()


def rule_match(answer: str, gold: str, aliases: list[str], min_chars: int = 2) -> bool:
    a = normalize(answer)
    if not a:
        return False
    for g in [gold, *aliases]:
        ng = normalize(g)
        if len(ng) >= min_chars and ng in a:
            return True
    return False


class JudgeOutput(BaseModel):
    verdict: Verdict
    reason: str = ""


def judge_answer(
    question: str, gold: str, aliases: list[str], answer: str, chat: ChatModel, prompt: Prompt | None = None
) -> dict[str, Any]:
    prompt = prompt or load_prompt("answer_judge")
    msgs = prompt.render(
        question=question, gold=gold, aliases="、".join(aliases) if aliases else "（无）", answer=answer
    )
    try:
        res = chat.chat(msgs, schema=JudgeOutput, prompt_version=prompt.version_id)
    except StructuredOutputError as e:
        return {"verdict": "wrong", "reason": f"judge 输出无法解析：{e}", "judge_error": True}
    out: JudgeOutput = res.parsed  # type: ignore[assignment]
    return {
        "verdict": out.verdict,
        "reason": out.reason,
        "judge_prompt": prompt.version_id,
        "usage": {"prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens},
    }


def score_answer(
    *,
    question: str,
    gold: str,
    aliases: list[str],
    answer: str,
    abstained: bool,
    judge: ChatModel | None,
    min_chars: int = 2,
) -> dict[str, Any]:
    """返回 {label, by, reason?}。label ∈ correct / partial / wrong / abstain；by ∈ abstain / rule / judge / none。"""
    if abstained:
        return {"label": "abstain", "by": "abstain"}
    if rule_match(answer, gold, aliases, min_chars):
        return {"label": "correct", "by": "rule"}
    if judge is None:
        return {"label": "wrong", "by": "none"}
    j = judge_answer(question, gold, aliases, answer, judge)
    return {"label": j["verdict"], "by": "judge", "reason": j.get("reason", ""), **({"judge_error": True} if
            j.get("judge_error") else {})}  # fmt: skip
