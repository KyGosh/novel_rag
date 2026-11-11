"""人物类别判定与关系清洗（P3.1，kg-schema-2）。

抽取口径里"人物"包括有名字的动物伙伴，还会混进少量器物、钱币、道具之类的非角色。消歧之后：
1. 类别判定：每个人物实体的名字、所属群体和若干条抽取描述成批交给 LLM，判为 人 / 动物 / 非角色 / 称呼
   （称呼交给 kg/epithet.py 做归属，归属不上的按"人"处理）。
   批次按实体 ID 排序后切分（确定性），LLM 结果按内容 hash 缓存。覆盖文件（aliases.toml 的 [[category]]）
   可以纠正个例。没有 LLM 结果（失败、漏项、不做判定）时默认是"人"。
2. 非角色实体连同它的提及、关系一起从图谱中去掉。
3. 关系清洗：按 schema 中每种关系的类别规则（schema.category_allows）检查两端的人物类别；动物与人之间的
   亲属、成员、效忠改写为 伙伴/坐骑（动物 → 人，证据合并保留），其余不允许的组合丢弃，按类型计数。
4. 常用词名字：名字写法本身是常用词（以外部常用词表为准，例如 IK 自带主词典），且语料里这个写法多数时候
   不是在指这个名字（被抽取为提及的次数 / 原文出现次数 < 阈值）时，记为"弱名字"。查询期链接器对弱名字只给
   弱匹配：固定流程不自动采用，Agent 的实体解析工具照样返回但会标注。

判定用的描述只在内部使用（可能含后续剧情），不进导出的实体，也不进 Neo4j。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from novel_rag.batch import run_jobs
from novel_rag.kg.names import normalize_name
from novel_rag.kg.schema import (
    ANIMAL,
    CATEGORY_JUDGMENTS,
    COMPANION_REWRITE_FROM,
    EPITHET,
    HUMAN,
    LOC,
    NON_CHARACTER,
    PERSON,
    category_allows,
)
from novel_rag.llm.base import ChatModel
from novel_rag.prompts import Prompt

CATEGORY_VERSION = "kg-category-4"  # 卡片内容、批次切分、默认值、清洗规则的版本


class CategoryItem(BaseModel):
    i: int
    category: str = Field(description="人 / 动物 / 非角色")
    reason: str = ""


class CategoryOut(BaseModel):
    items: list[CategoryItem] = Field(default_factory=list)


@dataclass(frozen=True)
class EntityCard:
    entity_id: str
    name: str
    aliases: tuple[str, ...]
    groups: tuple[str, ...]
    descriptions: tuple[str, ...]
    n_mentions: int

    def line(self, i: int) -> str:
        alias = "、".join(self.aliases) or "无"
        groups = "、".join(self.groups) or "无"
        descs = "；".join(self.descriptions) or "（无描述）"
        return f"{i}. {self.name}｜别名：{alias}｜群体：{groups}｜提及 {self.n_mentions} 次｜描述：{descs}"


@dataclass(frozen=True)
class Judgment:
    category: str
    reason: str
    source: str  # llm / default / override


def entity_cards(
    entities: Sequence[dict[str, Any]],
    m2e: Mapping[str, str],
    descriptions: Mapping[str, str],
    *,
    max_descriptions: int = 6,
    max_desc_chars: int = 60,
    max_aliases: int = 5,
) -> list[EntityCard]:
    """人物实体 → 判定卡片。描述按出现次数取最多的几条（类别按多数描述判，不被个别段落带偏），按实体 ID 排序。
    descriptions：提及 ID → 该提及的抽取描述。"""
    by_entity: dict[str, Counter] = defaultdict(Counter)
    for mid, eid in m2e.items():
        d = (descriptions.get(mid) or "").strip()
        if d:
            by_entity[eid][d[:max_desc_chars]] += 1
    cards = []
    for e in sorted((e for e in entities if e["type"] == PERSON), key=lambda e: e["entity_id"]):
        descs = by_entity.get(e["entity_id"], Counter())
        top = sorted(descs.items(), key=lambda x: (-x[1], x[0]))[:max_descriptions]
        aliases = [a["alias"] for a in e.get("aliases", []) if a["alias"] != e["name"]][:max_aliases]
        cards.append(
            EntityCard(
                e["entity_id"],
                e["name"],
                tuple(aliases),
                tuple(e.get("groups", [])[:3]),
                tuple(d for d, _ in top),
                int(e.get("n_mentions", 0)),
            )  # fmt: skip
        )
    return cards


def batches(cards: Sequence[EntityCard], size: int) -> list[list[EntityCard]]:
    return [list(cards[i : i + size]) for i in range(0, len(cards), max(1, size))]


async def judge_categories(
    cards: Sequence[EntityCard],
    chat: ChatModel,
    prompt: Prompt,
    *,
    batch_size: int,
    concurrency: int,
    stats: Counter,
) -> dict[str, Judgment]:
    """成批判定；失败的批次、漏判或判成未知类别的项默认为"人"（并计数）。"""

    async def run(batch: list[EntityCard]) -> CategoryOut:
        items = "\n".join(c.line(i) for i, c in enumerate(batch, 1))
        res = await chat.achat(
            prompt.render(n=str(len(batch)), items=items), schema=CategoryOut, prompt_version=prompt.version_id
        )
        return res.parsed  # type: ignore[return-value]

    out: dict[str, Judgment] = {}
    jobs = batches(cards, batch_size)
    stats["category_batches"] = len(jobs)
    results = await run_jobs(jobs, run, concurrency=concurrency, label="kg_category")
    for r in results:
        batch: list[EntityCard] = r.job
        got: dict[int, CategoryItem] = {}
        if r.ok and r.value is not None:
            for it in r.value.items:
                got.setdefault(it.i, it)
        else:
            stats["category_batch_failed"] += 1
        for i, c in enumerate(batch, 1):
            it = got.get(i)
            cat = (it.category or "").strip() if it else ""
            if cat in CATEGORY_JUDGMENTS:
                out[c.entity_id] = Judgment(cat, (it.reason if it else "")[:40], "llm")
            else:
                stats["category_missing" if it is None else "category_invalid"] += 1
                out[c.entity_id] = Judgment(HUMAN, "", "default")
    return out


def place_name_rule(judgments: dict[str, Judgment], entities: Sequence[dict[str, Any]], stats: Counter) -> None:
    """地名转指（用地名指代该地的领主或居民）：被判为"称呼"的人物实体，规范名（不含间隔号）与某个地点实体的
    名字或别名相同时，改判"非角色"，不做称呼归属。只作用于"称呼"：判为"人"的同名实体多是碰巧与地名同名的
    真人（P3.1 试过对"人"也生效，误删了大量真人）。在覆盖文件之前执行，覆盖文件可以改回。"""
    places = set()
    for e in entities:
        if e["type"] == LOC:
            places.update(normalize_name(n) for n in [e["name"], *(a["alias"] for a in e.get("aliases", []))])
    for e in entities:
        n = normalize_name(e["name"])
        j = judgments.get(e["entity_id"])
        if e["type"] == PERSON and "·" not in n and n in places and j is not None and j.category == EPITHET:
            judgments[e["entity_id"]] = Judgment(NON_CHARACTER, "与地点同名（地名转指）", "rule")
            stats["category_place_name"] += 1


def apply_category_overrides(
    judgments: dict[str, Judgment],
    entities: Sequence[dict[str, Any]],
    overrides: Mapping[str, str],
    stats: Counter,
) -> None:
    """覆盖文件：键是实体 ID 或规范名（同名的人物实体都改），值是类别。"""
    by_name: dict[str, list[str]] = defaultdict(list)
    ids = set()
    for e in entities:
        if e["type"] == PERSON:
            by_name[normalize_name(e["name"])].append(e["entity_id"])
            ids.add(e["entity_id"])
    for key, cat in overrides.items():
        if cat not in CATEGORY_JUDGMENTS:
            raise ValueError(f"aliases.toml 的类别覆盖 {key!r} = {cat!r} 不合法（可选 {CATEGORY_JUDGMENTS}）")
        targets = [key] if key in ids else by_name.get(normalize_name(key), [])
        if not targets:
            stats["category_override_unresolved"] += 1
            continue
        for eid in targets:
            judgments[eid] = Judgment(cat, "覆盖文件", "override")
            stats["category_override_applied"] += 1


def assign_categories(entities: list[dict[str, Any]], judgments: Mapping[str, Judgment]) -> set[str]:
    """把类别写进人物实体（没有判定的、判为"称呼"而没被归属掉的都按"人"）；返回判为非角色的实体 ID。"""
    dropped = set()
    for e in entities:
        if e["type"] != PERSON:
            continue
        j = judgments.get(e["entity_id"])
        cat = j.category if j else HUMAN
        if cat == EPITHET:
            cat = HUMAN
        if cat == NON_CHARACTER:
            dropped.add(e["entity_id"])
        e["category"] = cat if cat != NON_CHARACTER else None
    return dropped


def drop_entities(
    entities: Sequence[dict[str, Any]], m2e: Mapping[str, str], drop: set[str], stats: Counter
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    kept = [e for e in entities if e["entity_id"] not in drop]
    new_m2e = {m: e for m, e in m2e.items() if e not in drop}
    stats["non_character_entities_dropped"] = len(entities) - len(kept)
    stats["non_character_mentions_dropped"] = len(m2e) - len(new_m2e)
    return kept, new_m2e


def person_category_of(entities: Iterable[dict[str, Any]]) -> dict[str, str | None]:
    """实体 ID → 人物类别；非人物实体为 None（关系类别规则不约束这一端）。"""
    return {e["entity_id"]: (e.get("category") or HUMAN) if e["type"] == PERSON else None for e in entities}


def _merge_evidence(evs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for e in sorted(
        evs, key=lambda e: (e["reading_order"], e["chunk_id"], e["start"] if e["start"] is not None else -1)
    ):
        k = (e["chunk_id"], e["start"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def clean_relations(
    relations: Sequence[dict[str, Any]],
    category_of: Mapping[str, str | None],
    dropped: set[str],
    stats: Counter,
) -> list[dict[str, Any]]:
    """按关系类型的类别规则清洗聚合后的关系（逐块抽取的关系与事件关系都适用）：
    - 牵涉非角色实体的关系去掉；
    - 动物与人之间的 亲属 / 成员 / 效忠 改写为 伙伴/坐骑（动物 → 人），同一对的多条合并，证据全部保留，
      rewritten_from 记录来源（类型:角色 → 证据数）；
    - 其他不允许的类别组合丢弃，按类型计数；允许的原样保留（动物之间的亲属保留）。"""
    kept: list[dict[str, Any]] = []
    companion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sources: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in relations:
        s, o, t = r["subject"], r["object"], r["type"]
        if s in dropped or o in dropped:
            stats[f"rel_dropped_non_character_{t}"] += 1
            continue
        sc, oc = category_of.get(s), category_of.get(o)
        if category_allows(t, sc, oc):
            kept.append(r)
            continue
        if t in COMPANION_REWRITE_FROM and {sc, oc} == {ANIMAL, HUMAN}:
            animal, human = (s, o) if sc == ANIMAL else (o, s)
            companion[(animal, human)].extend(r["evidence"])
            sources[(animal, human)][f"{t}:{r['role']}" if r.get("role") else t] += r["n_evidence"]
            stats[f"rel_rewritten_companion_from_{t}"] += 1
            continue
        stats[f"rel_dropped_category_{t}"] += 1
    for (animal, human), evs in sorted(companion.items()):
        ev = _merge_evidence(evs)
        kept.append(
            {
                "subject": animal,
                "type": "COMPANION",
                "role": None,
                "object": human,
                "first_reading_order": ev[0]["reading_order"],
                "last_reading_order": ev[-1]["reading_order"],
                "n_evidence": len(ev),
                "evidence": ev,
                "rewritten_from": dict(sorted(sources[(animal, human)].items())),
            }
        )
    stats["rel_companion"] = len(companion)
    return kept


# ---- 常用词名字 ----


def load_word_list(path: Path | None) -> set[str] | None:
    """常用词表：每行一个词（IK 词典格式）。文件不存在时返回 None（不做常用词判定）。"""
    if path is None or not path.is_file():
        return None
    words = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip().lstrip("﻿")
        if w:
            words.add(w)
    return words


def weak_names(
    entities: Sequence[dict[str, Any]],
    linked_counts: Mapping[str, int],
    corpus_count: Any,
    common_words: set[str],
    *,
    min_linked_ratio: float,
    stats: Counter,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """实体 ID → 该实体的弱名字写法（规范名或别名）。

    写法是常用词，且"被抽取为实体提及的次数 / 在原文中出现的次数" < min_linked_ratio 时为弱名字：
    "夏天""淑女"在原文里多数时候是普通词；而"山姆""君临"虽然也在词表里，原文里几乎总是指这个名字，不算弱。
    linked_counts：写法 → 被抽取为提及的次数；corpus_count(写法, 更长的名字) → 原文出现次数，不算出现在更长的
    实体名字里的（"凯特"不算"凯特琳"里的那次，"里斯"不算"克里斯"里的）。返回 (映射, 明细)。"""
    all_forms = set()
    for e in entities:
        for f in [e["name"], *(a["alias"] for a in e.get("aliases", []))]:
            n = normalize_name(f)
            all_forms.update({n, n.replace("·", "")})
    ratio_of: dict[str, tuple[int, int, float]] = {}
    out: dict[str, list[str]] = {}
    for e in entities:
        forms = [e["name"], *(a["alias"] for a in e.get("aliases", []))]
        weak = []
        for f in dict.fromkeys(forms):
            w = normalize_name(f)
            if "·" in w or w not in common_words:
                continue
            if w not in ratio_of:
                longer = sorted((f for f in all_forms if w in f and f != w), key=len, reverse=True)
                total = int(corpus_count(w, longer))
                linked = int(linked_counts.get(w, 0))
                ratio_of[w] = (linked, total, linked / total if total else 1.0)
            if ratio_of[w][2] < min_linked_ratio:
                weak.append(f)
        if weak:
            out[e["entity_id"]] = weak
    details = [
        {"form": w, "linked": a, "corpus": b, "ratio": round(r, 3), "weak": r < min_linked_ratio}
        for w, (a, b, r) in sorted(ratio_of.items(), key=lambda x: (x[1][2], x[0]))
    ]
    stats["common_word_forms"] = len(ratio_of)
    stats["weak_forms"] = sum(1 for d in details if d["weak"])
    stats["entities_with_weak_names"] = len(out)
    return out, details


def linked_form_counts(records: Sequence[dict[str, Any]]) -> Counter:
    """写法 → 被抽取为实体提及的总次数（任何类型；提及的 count 是该写法在本段出现的次数）。"""
    c: Counter = Counter()
    for rec in records:
        for m in rec["mentions"]:
            c[normalize_name(m["name"])] += int(m.get("count", 1))
    return c


class CorpusCounter:
    """原文出现次数：与抽取时提及计数同一口径（对每个 chunk 做 normalize_name 后数子串，chunk 重叠部分两边都算）。"""

    def __init__(self, texts: Iterable[str]):
        self.blob = "\x00".join(normalize_name(t) for t in texts)

    def __call__(self, word: str, longer: Sequence[str] = ()) -> int:
        """word 的出现次数；longer 中的更长写法先遮掉（从长到短），出现在它们里面的不算。"""
        if not word:
            return 0
        blob = self.blob
        for f in longer:
            if f in blob:
                blob = blob.replace(f, "\x01")
        return blob.count(word)
