"""逐 chunk 的实体、关系、别名抽取与校验。

每个 chunk 调用一次 LLM；上下文给卷名、章节标题、POV，以及本 chunk 之前的一小段原文（标注为"仅供判断指代"）。
模型输出先过校验，不合格的条目丢弃并按原因计数：
- 实体：类型不合法 / 扩展未启用 / 名字不在本段原文中 / 泛称；全名既与名字不相干、又不出现在本段和上文时，
  不采信（记为 claimed_full_name，留给消歧阶段当候选线索），防止模型凭自带知识补全身份。
- 关系：主客体不是本段实体 / 说法映射不到白名单 / 主客体类型不符 / 证据引文在本段定位不到（防止编造关系的主要手段）。
- 别名：别称不在本段原文中 / 证据引文定位不到。
校验只依赖原始输出，可以在缓存的输出上重跑。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from novel_rag.corpus import BookText
from novel_rag.eval.qgen.locate import locate
from novel_rag.kg.names import NameRules, normalize_name, segments
from novel_rag.kg.schema import (
    ORG,
    ORG_CATEGORIES,
    REL_BY_NAME,
    TITLE,
    TYPE_COERCIONS,
    Schema,
    entity_type_of,
    map_relation,
)
from novel_rag.llm.base import ChatModel, Message
from novel_rag.prompts import Prompt

VALIDATION_VERSION = "kgx-val-3"
SECTION_NOTES = {
    "appendix": "\n（本段来自附录中的人物名表：逐条抽取其中写明的亲属、婚姻、成员、效忠关系；"
    "表中「某某之子 / 之女」的写法即亲属关系。）"
}


class EntityOut(BaseModel):
    name: str = Field(description="本段原文中的写法")
    full_name: str | None = Field(default=None, description="原文给出的全名；不能确定时为 null")
    type: str = Field(description="人物 / 组织 / 地点（/ 头衔）")
    category: str | None = Field(default=None, description="组织的类别")
    group: str | None = Field(default=None, description="人物所属的家族或组织")
    description: str = Field(default="", description="一句话消歧描述")


class RelationOut(BaseModel):
    subject: str
    relation: str
    role: str | None = None
    object: str
    evidence: str = Field(default="", description="逐字摘自本段原文的引文，不超过 60 字")


class AliasOut(BaseModel):
    entity: str
    alias: str
    evidence: str = ""


class ChunkExtractionOut(BaseModel):
    entities: list[EntityOut] = Field(default_factory=list)
    relations: list[RelationOut] = Field(default_factory=list)
    aliases: list[AliasOut] = Field(default_factory=list)


def extract_version(schema: Schema, prompt: Prompt, model_identity: dict[str, Any], chunk_version: str) -> str:
    """决定 LLM 原始输出的因素（校验规则不在内：改校验只需 --revalidate，不必重新调用 LLM）。"""
    schema_text = schema.prompt_entity_types() + schema.prompt_relation_types()  # schema 说明是 prompt 的一部分
    payload = {
        "schema": schema.version,
        "schema_prompt": hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:12],
        "prompt": prompt.version_id,
        "model": model_identity,
        "chunk_version": chunk_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def chunk_context(book: BookText, chunk: dict[str, Any], width: int) -> str:
    """本 chunk 之前、同一章内的一小段原文（与上一个 chunk 的重叠部分本来就在本段里，这里取本段起点之前的文字）。"""
    ch = book.chapter(chunk["chapter_idx"])
    start = max(ch.body_start, chunk["char_start"] - width)
    return book.text[start : chunk["char_start"]].strip()


def build_messages(
    book: BookText, chunk: dict[str, Any], schema: Schema, prompt: Prompt, context_chars: int
) -> list[Message]:
    ctx = chunk_context(book, chunk, context_chars)
    return prompt.render(
        entity_types=schema.prompt_entity_types(),
        relation_types=schema.prompt_relation_types(),
        book_title=book.title or book.book_id,
        chapter_title=chunk.get("chapter_title") or "",
        pov=chunk.get("pov") or "无",
        section_note=SECTION_NOTES.get(chunk.get("section_type", ""), ""),
        context=ctx or "（无，本段是章节开头）",
        text=chunk["text"],
    )


def _squash(s: str) -> str:
    return normalize_name(s).replace("·", "")


def appears(name: str, text: str) -> bool:
    """名字是否出现在文本中（统一间隔号、忽略空白；也接受原文省略了间隔号的写法）。"""
    n = normalize_name(name)
    if not n:
        return False
    t = normalize_name(text)
    if n in t or (("·" in n) and _squash(n) in _squash(t)):
        return True
    # 组织名被补上了"家族 / 家"后缀（原文只写"某某"或"某某家"）
    return any(n.endswith(suf) and len(n) - len(suf) >= 2 and n[: -len(suf)] in t for suf in ("家族", "家"))


def full_name_supported(name: str, full: str, text: str, context: str, rules: NameRules) -> bool:
    """全名可以采信：与名字相同或有共同片段（"琼恩" ⊂ "琼恩·雪诺"），或全名的某个片段出现在本段 / 上文中。"""
    fn = normalize_name(full)
    nm = rules.strip_honorifics(name)
    if not fn:
        return False
    if fn == normalize_name(name) or nm in fn:
        return True
    fsegs = segments(fn)
    if set(segments(nm)) & set(fsegs):
        return True
    hay = normalize_name(text + context)
    return any(len(s) >= 2 and s in hay for s in fsegs)


def validate_extraction(
    out: ChunkExtractionOut,
    book: BookText,
    chunk: dict[str, Any],
    context: str,
    schema: Schema,
    rules: NameRules,
    *,
    fuzzy_threshold: float,
) -> tuple[dict[str, Any], Counter]:
    text = chunk["text"]
    drops: Counter = Counter()
    mentions: list[dict[str, Any]] = []
    by_name: dict[str, int] = {}
    seen: dict[tuple[str, str], int] = {}

    def add_mention(
        name: str, full: str | None, claimed: str | None, etype: str, category: str | None, group: str | None,
        description: str, implicit: bool,
    ) -> int:  # fmt: skip
        key_name = rules.canonical(full or name)
        dkey = (etype, key_name)
        if dkey in seen:  # 同一实体在本段重复列出
            drops["entity_duplicate"] += 1
            return seen[dkey]
        if etype == ORG and category not in ORG_CATEGORIES:
            category = "其他" if category else None
        idx = len(mentions)
        seen[dkey] = idx
        mentions.append(
            {
                "m": idx,
                "name": name,
                "full_name": full,
                "claimed_full_name": claimed,
                "key": key_name,
                "type": etype,
                "category": category if etype == ORG else None,
                "group": normalize_name(group or "") or None,
                "description": (description or "").strip()[:60],
                "count": max(1, normalize_name(text).count(name)),
                "implicit": implicit,
            }
        )
        for alias in (name, full, key_name):
            if alias:
                by_name.setdefault(normalize_name(alias), idx)
                by_name.setdefault(rules.canonical(alias), idx)
        return idx

    for e in out.entities:
        etype, cat_from_type = entity_type_of(e.type)
        if etype is None or etype not in schema.entity_types:
            drops["entity_extension_disabled" if etype == TITLE else "entity_bad_type"] += 1
            continue
        name = normalize_name(e.name)
        if not appears(name, text):
            if e.full_name and appears(e.full_name, text):
                name = normalize_name(e.full_name)
            else:
                drops["entity_name_not_in_text"] += 1
                continue
        full, claimed = None, None
        if e.full_name and normalize_name(e.full_name) != name:
            if full_name_supported(name, e.full_name, text, context, rules):
                full = normalize_name(e.full_name)
            else:
                claimed = normalize_name(e.full_name)
                drops["full_name_unverified"] += 1  # 实体保留，只是不采信全名
        # 泛称（"国王""王后"）只有在原文支持的全名下才保留，键用全名
        if etype != TITLE and rules.is_generic(name) and (full is None or rules.is_generic(full)):
            drops["entity_generic"] += 1
            continue
        idx = add_mention(
            name, full, claimed, etype, (e.category or "").strip() or cat_from_type, e.group, e.description, False
        )
        if e.name:
            by_name.setdefault(normalize_name(e.name), idx)

    def implicit_entity(n: str, rel_type: str, side: str) -> int | None:
        """关系端点没有列在 entities 里：名字出现在本段、不是泛称、能由关系类型约束推出实体类型时补上。"""
        n = normalize_name(n)
        if not appears(n, text) or rules.is_generic(n):
            return None
        rt = REL_BY_NAME[rel_type]
        allowed = [t for t in (rt.subject_types if side == "s" else rt.object_types) if t in schema.entity_types]
        if len(allowed) > 1:
            org_like = n.endswith(("家族", "家", "军", "会", "团", "卫", "教"))
            allowed = [ORG] if org_like and ORG in allowed else ([t for t in allowed if t != ORG] or allowed)
        if len(allowed) != 1:
            return None
        drops["entity_implicit"] += 1
        return add_mention(n, None, None, allowed[0], None, None, "", True)

    def resolve(n: str) -> int | None:
        n = normalize_name(n)
        return by_name.get(n, by_name.get(rules.canonical(n)))

    def find_quote(q: str) -> tuple[int, int, str] | None:
        if not q or not q.strip():
            return None
        loc = locate(book.text, q, chunk["char_start"], chunk["char_end"], fuzzy_threshold=fuzzy_threshold)
        return (loc.start, loc.end, loc.method) if loc else None

    relations: list[dict[str, Any]] = []
    rel_seen: set[tuple] = set()
    for r in out.relations:
        mapped = map_relation(r.relation, r.role)
        if mapped is None:
            drops["relation_unmapped"] += 1
            continue
        if not schema.allows(mapped.type):
            drops["relation_extension_disabled"] += 1
            continue
        s_name, o_name = (r.object, r.subject) if mapped.flip else (r.subject, r.object)
        si, oi = resolve(s_name), resolve(o_name)
        if si is None:
            si = implicit_entity(s_name, mapped.type, "s")
        if oi is None:
            oi = implicit_entity(o_name, mapped.type, "o")
        if si is None or oi is None:
            drops["relation_unknown_entity"] += 1
            continue
        if si == oi:
            drops["relation_self"] += 1
            continue
        rel_type = TYPE_COERCIONS.get((mapped.type, mentions[oi]["type"]), mapped.type)
        if rel_type != mapped.type:
            drops["relation_coerced"] += 1
            if not schema.allows(rel_type):
                drops["relation_extension_disabled"] += 1
                continue
        role = mapped.role if rel_type == "KIN" else None
        rt = REL_BY_NAME[rel_type]
        st, ot = mentions[si]["type"], mentions[oi]["type"]
        if st not in rt.subject_types or ot not in rt.object_types:
            if rt.symmetric and ot in rt.subject_types and st in rt.object_types:
                si, oi = oi, si
            else:
                drops["relation_type_mismatch"] += 1
                continue
        quote = find_quote(r.evidence)
        if quote is None:
            drops["relation_evidence_not_found"] += 1
            continue
        key = (si, rel_type, role, oi)
        if key in rel_seen:
            drops["relation_duplicate"] += 1
            continue
        rel_seen.add(key)
        relations.append(
            {"s": si, "type": rel_type, "role": role, "o": oi, "evidence": list(quote[:2]), "match": quote[2]}
        )

    aliases: list[dict[str, Any]] = []
    for a in out.aliases:
        idx = resolve(a.entity)
        alias = normalize_name(a.alias)
        if idx is None:
            drops["alias_unknown_entity"] += 1
            continue
        if not alias or alias in (mentions[idx]["name"], mentions[idx]["full_name"], mentions[idx]["key"]):
            drops["alias_trivial"] += 1
            continue
        if not appears(alias, text):
            drops["alias_not_in_text"] += 1
            continue
        if rules.is_generic(alias):
            drops["alias_generic"] += 1
            continue
        quote = find_quote(a.evidence)
        if quote is None:
            drops["alias_evidence_not_found"] += 1
            continue
        aliases.append({"m": idx, "alias": alias, "evidence": list(quote[:2]), "match": quote[2]})

    record = {
        "chunk_id": chunk["chunk_id"],
        "book_id": chunk["book_id"],
        "chapter_idx": chunk["chapter_idx"],
        "reading_order": chunk["reading_order"],
        "section_type": chunk["section_type"],
        "mentions": mentions,
        "relations": relations,
        "aliases": aliases,
        "drops": dict(drops),
        "validation_version": VALIDATION_VERSION,
    }
    return record, drops


async def extract_chunk(
    book: BookText,
    chunk: dict[str, Any],
    chat: ChatModel,
    prompt: Prompt,
    schema: Schema,
    rules: NameRules,
    *,
    context_chars: int,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    msgs = build_messages(book, chunk, schema, prompt, context_chars)
    result = await chat.achat(msgs, schema=ChunkExtractionOut, prompt_version=prompt.version_id)
    out = result.parsed
    assert isinstance(out, ChunkExtractionOut)
    context = chunk_context(book, chunk, context_chars)
    record, _ = validate_extraction(out, book, chunk, context, schema, rules, fuzzy_threshold=fuzzy_threshold)
    record["raw"] = out.model_dump()
    record["llm"] = {
        "model": result.model,
        "cached": result.cached,
        "attempts": result.attempts,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "latency_s": round(result.latency_s, 2),
    }
    return record


def revalidate(
    records: Sequence[dict[str, Any]],
    books: dict[str, BookText],
    chunks_by_id: dict[str, dict[str, Any]],
    schema: Schema,
    rules: NameRules,
    *,
    context_chars: int,
    fuzzy_threshold: float,
) -> list[dict[str, Any]]:
    """在已保存的原始输出上重跑校验（改校验规则后不必重新调用 LLM）。"""
    out = []
    for rec in records:
        chunk = chunks_by_id[rec["chunk_id"]]
        book = books[chunk["book_id"]]
        raw = ChunkExtractionOut.model_validate(rec["raw"])
        ctx = chunk_context(book, chunk, context_chars)
        new, _ = validate_extraction(raw, book, chunk, ctx, schema, rules, fuzzy_threshold=fuzzy_threshold)
        new["raw"], new["llm"] = rec["raw"], rec["llm"]
        out.append(new)
    return out
