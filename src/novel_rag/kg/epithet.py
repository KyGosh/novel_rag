"""称呼归属（P3.1 第 7 项）：把绰号、自称、代称这类孤立的人物簇并进它所指的真实人物，作为别名。

这类称呼在抽取时没有全名、原文也没有"人称 X"式的别名陈述，消歧的三条合并途径都用不上。类别判定把它们
标为"称呼"之后：
1. 取称呼最早出现的若干段（按阅读顺序），每段截取称呼前后的原文窗口；候选人物是这些段所在 chunk 和所在章
   出现过的人物（不含非角色和其他称呼），别名只列在这些段之前已经可见的。
2. LLM 逐段判断称呼指候选中的哪一位，只能依据段落文字，拿不准填 0。结果按内容 hash 缓存。
3. 代码复核（宁可少并，不要错并）：
   - 目标必须在该段所在的章出现过（候选表之外的编号、不在同章的目标都作废）；
   - LLM 自检 same_person 为假（选的是与称呼所指之人相关的人，而不是他本人）的段作废；
   - "某人的某某"：称呼里含"目标的名字或别名 + 的"时放弃（称呼指的是某人的随从、亲属等，不是某人）；
   - 真名保护：称呼本身含间隔号（是全名），或是任何全名人物的名或姓，不合并；
   - 各段指向不同的人时放弃；指向同一人的段数 / 判定段数 < min_support 时放弃。
4. 专属称呼与泛称分开（generic_mode="link"，默认）：
   - LLM 同时判断称呼本身是"专属"（只指一个特定的人）还是"泛称"（身份、职务、亲属称谓、描述语）。
     判为泛称、各段指向不同的人、或有一段明确指候选之外的人，都按泛称处理。
   - 专属称呼：复核通过后作为目标的别名，从第一次被归属的那一段起可见（防剧透）；改挂"有段落被归属到目标"
     的那些章里、不早于可见起点的提及，其余留在称呼实体上。归属不上的专属称呼（真名不明的外号）保留为
     孤立的人物实体。
   - 泛称：不作为任何人的别名，只按章挂靠——某章被判定的段落都指同一人（且该章没有段落指候选之外的人），
     这一章里该称呼的提及改挂给他。有证据指向不同的人时（multi），没挂上的提及连同称呼实体一起删掉，
     不留一个把多人串在一起的假人物；没有这种证据的（例如只以身份为人所知的人），剩余提及保留为孤立人物。
   generic_mode="alias" 是 fd032ef1 的旧行为：不区分泛称，各段指向不同的人时放弃，归属成功的只改挂
   "有段落被归属到目标"的那些章里的提及，并作为别名。
   目标实体的计数和首末位置重算。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from novel_rag.batch import run_jobs
from novel_rag.kg.names import normalize_name, segments
from novel_rag.kg.schema import EPITHET, NON_CHARACTER, PERSON
from novel_rag.llm.base import ChatModel
from novel_rag.prompts import Prompt

EPITHET_VERSION = "kg-epithet-5"  # 段落选取、候选构造、复核与合并规则的版本


class MentionLike(Protocol):
    mid: str
    chunk_id: str
    book_id: str
    chapter_idx: int
    reading_order: int
    name: str
    count: int


class PassageVerdict(BaseModel):
    p: int
    target: int = Field(default=0, description="候选编号；无法确定为 0")
    same_person: bool = Field(default=False, description="所选候选就是称呼所指的那个人本身")
    other_person: bool = Field(default=False, description="target 为 0，但段落明确指候选之外的某个具体的人")
    basis: str = ""


class EpithetOut(BaseModel):
    kind: Literal["专属", "泛称"] = Field(default="泛称", description="称呼本身是专属称呼还是泛称")
    passages: list[PassageVerdict] = Field(default_factory=list)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    book_id: str
    chapter_idx: int
    reading_order: int
    text: str  # 称呼前后的原文窗口，称呼用【】标出


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    display: str


@dataclass
class EpithetJob:
    entity_id: str
    name: str
    passages: list[Passage]
    candidates: list[Candidate]


@dataclass
class Attribution:
    epithet_id: str
    name: str
    target_id: str | None
    # merged（专属称呼并入，作别名）/ linked（泛称按章挂靠）/ generic_unlinked（泛称，一章也没挂上）/
    # no_target / conflict / low_support / real_name / possessive / no_passage / llm_failed
    status: str
    visible_from: int | None = None
    first_chunk: str | None = None
    votes: dict[str, int] = field(default_factory=dict)
    n_passages: int = 0
    basis: list[str] = field(default_factory=list)
    chapters: list[list[Any]] = field(default_factory=list)  # 有段落被归属到目标的章：[book_id, chapter_idx]
    kind: str | None = None  # LLM 判的"专属"/"泛称"
    generic: bool = False  # 最终是否按泛称处理
    all_chapters: bool = False  # 专属称呼：从可见起点起改挂全部提及（不限于被归属的章）
    links: list[list[Any]] = field(default_factory=list)  # 泛称按章挂靠：[book_id, chapter_idx, 目标 ID, 依据]
    multi: bool = False  # 泛称且有证据指向不同的人（各段目标不同，或有段落指候选之外的人）
    # 逐段原始判定：[chunk_id, 目标 ID|None, 本人, 指他人, 依据]
    verdicts: list[list[Any]] = field(default_factory=list)


def find_raw(text: str, name: str) -> tuple[int, int] | None:
    """name（规范化写法）在原文中的区间。规范化会去掉引号和空白，位置会错开，所以先在原文里直接找；
    找不到时在规范化文本里找，再把位置映射回原文。"""
    if not name:
        return None
    i = text.find(name)
    if i >= 0:
        return i, i + len(name)
    keep = [k for k, ch in enumerate(text) if normalize_name(ch)]
    j = "".join(text[k] for k in keep).find(name)
    if j < 0:
        return None
    return keep[j], keep[j + len(name) - 1] + 1


def _window(text: str, name: str, chars: int) -> str | None:
    span = find_raw(text, name)
    if span is None:
        return None
    i, j = span
    return (text[max(0, i - chars) : i] + "【" + text[i:j] + "】" + text[j : j + chars]).replace("\n", " ")


def build_jobs(
    entities: Sequence[dict[str, Any]],
    m2e: Mapping[str, str],
    mentions: Mapping[str, MentionLike],
    chunk_text: Mapping[str, str],
    categories: Mapping[str, str],
    *,
    max_passages: int,
    window_chars: int,
    max_candidates: int,
) -> list[EpithetJob]:
    """categories：实体 ID → 判定类别。只为判为"称呼"的人物实体建任务，按实体 ID 排序（确定性）。"""
    by_id = {e["entity_id"]: e for e in entities}
    mids_of: dict[str, list[str]] = defaultdict(list)
    in_chunk: dict[str, set[str]] = defaultdict(set)
    in_chapter: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for mid, eid in m2e.items():
        mids_of[eid].append(mid)
        m = mentions[mid]
        in_chunk[m.chunk_id].add(eid)
        in_chapter[(m.book_id, m.chapter_idx)][eid] += m.count

    def eligible(eid: str) -> bool:
        e = by_id.get(eid)
        return e is not None and e["type"] == PERSON and categories.get(eid) not in (NON_CHARACTER, EPITHET)

    jobs = []
    for e in sorted(entities, key=lambda e: e["entity_id"]):
        eid = e["entity_id"]
        if e["type"] != PERSON or categories.get(eid) != EPITHET:
            continue
        ms = sorted((mentions[m] for m in mids_of[eid]), key=lambda m: (m.reading_order, m.chunk_id, m.mid))
        passages: list[Passage] = []
        seen_chunks: set[str] = set()
        for m in ms:
            if m.chunk_id in seen_chunks or len(passages) >= max_passages:
                continue
            w = _window(chunk_text.get(m.chunk_id, ""), m.name, window_chars)
            if w is None:
                continue
            seen_chunks.add(m.chunk_id)
            passages.append(Passage(m.chunk_id, m.book_id, m.chapter_idx, m.reading_order, w))
        if not passages:
            jobs.append(EpithetJob(eid, e["name"], [], []))
            continue
        # 候选：段内出现的人物优先，其次是同章按提及数排序
        score: Counter = Counter()
        for p in passages:
            for c in in_chunk[p.chunk_id]:
                if c != eid and eligible(c):
                    score[c] += 1000
            for c, n in in_chapter[(p.book_id, p.chapter_idx)].items():
                if c != eid and eligible(c):
                    score[c] += n
        first_ro = passages[0].reading_order
        cands = []
        for c, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:max_candidates]:
            ce = by_id[c]
            vis = [a["alias"] for a in ce.get("aliases", []) if a["visible_from"] <= first_ro][:3]
            name = ce["name"] if ce.get("name_visible_from", 0) <= first_ro or not vis else vis[0]
            others = [a for a in vis if a != name]
            cands.append(Candidate(c, name + (f"〔{'、'.join(others)}〕" if others else "")))
        jobs.append(EpithetJob(eid, e["name"], passages, cands))
    return jobs


def render_job(job: EpithetJob, prompt: Prompt) -> list[dict[str, str]]:
    cands = "\n".join(f"{i}. {c.display}" for i, c in enumerate(job.candidates, 1)) or "（无）"
    passages = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(job.passages, 1))
    return prompt.render(epithet=job.name, candidates=cands, passages=passages)


def full_name_segments(entities: Sequence[dict[str, Any]]) -> set[str]:
    """所有含间隔号的人物名字（规范名与别名）的名和姓片段，供真名保护使用。"""
    segs: set[str] = set()
    for e in entities:
        if e["type"] != PERSON:
            continue
        for n in [e["name"], *(a["alias"] for a in e.get("aliases", []))]:
            n = normalize_name(n)
            if "·" in n:
                segs.update(segments(n))
    return segs


def name_forms(entities: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """实体 ID → 名字写法（规范名、别名，以及含间隔号写法的各片段，只取 2 字以上的）。"""
    out: dict[str, set[str]] = {}
    for e in entities:
        forms: set[str] = set()
        for n in [e["name"], *(a["alias"] for a in e.get("aliases", []))]:
            n = normalize_name(n)
            forms.add(n)
            forms.update(segments(n))
        out[e["entity_id"]] = {f for f in forms if len(f) >= 2}
    return out


def decide(
    job: EpithetJob,
    out: EpithetOut | None,
    mentions_by_chapter: Mapping[tuple[str, int], set[str]],
    real_name_segments: set[str],
    *,
    min_support: float,
    forms_of: Mapping[str, set[str]] | None = None,
    generic_mode: str = "link",
) -> Attribution:
    att = Attribution(job.entity_id, job.name, None, "no_target", n_passages=len(job.passages))
    name = normalize_name(job.name)
    if "·" in name or name in real_name_segments:
        att.status = "real_name"
        return att
    if not job.passages:
        att.status = "no_passage"
        return att
    if out is None:
        att.status = "llm_failed"
        return att
    got = {v.p: v for v in out.passages}
    att.kind = out.kind
    for i, p in enumerate(job.passages, 1):
        v = got.get(i)
        if v is not None:
            tid0 = job.candidates[v.target - 1].entity_id if 1 <= v.target <= len(job.candidates) else None
            att.verdicts.append([p.chunk_id, tid0, v.same_person, v.other_person, v.basis[:20]])
    forms = forms_of or {}

    def possessive(tid: str) -> bool:
        return any(f + "的" in name for f in forms.get(tid, ()))

    other_chapters = {
        (p.book_id, p.chapter_idx)
        for i, p in enumerate(job.passages, 1)
        if (v := got.get(i)) is not None and v.other_person and not v.target
    }
    votes: Counter = Counter()
    first: dict[str, Passage] = {}
    attributed: dict[str, list[Passage]] = defaultdict(list)
    basis: dict[str, list[str]] = defaultdict(list)
    for i, p in enumerate(job.passages, 1):
        v = got.get(i)
        if v is None or not 1 <= v.target <= len(job.candidates) or not v.same_person:
            continue
        tid = job.candidates[v.target - 1].entity_id
        if tid not in mentions_by_chapter.get((p.book_id, p.chapter_idx), set()):
            continue  # 目标不在该段所在的章：作废
        votes[tid] += 1
        first.setdefault(tid, p)
        attributed[tid].append(p)
        basis[tid].append(v.basis[:20])
    att.votes = dict(votes)
    # 指向不同的人的证据：各段目标不同；或有段落指候选之外的人、同时另有段落指某个候选。
    # 只有"指候选之外的人"不算——那通常就是称呼的主人自己不在候选里（真名不明的外号）。
    multi = len(votes) > 1 or bool(votes and other_chapters)
    if generic_mode == "link" and (out.kind == "泛称" or multi):
        att.generic, att.multi = True, multi
        per_chapter: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        for tid, ps in attributed.items():
            for p, b in zip(ps, basis[tid], strict=True):
                ch = (p.book_id, p.chapter_idx)
                if not per_chapter[ch].get(tid):
                    per_chapter[ch][tid] = b
        for ch in sorted(per_chapter):
            tids = per_chapter[ch]
            if len(tids) != 1 or ch in other_chapters:
                continue  # 同一章内指向不一致，或有段落指候选之外的人：这一章不挂
            tid, b = next(iter(tids.items()))
            if possessive(tid):
                continue
            att.links.append([ch[0], ch[1], tid, b])
        att.status = "linked" if att.links else "generic_unlinked"
        return att
    if not votes:
        return att
    if len(votes) > 1:
        att.status = "conflict"
        return att
    tid, n = votes.most_common(1)[0]
    if possessive(tid):
        att.status = "possessive"
        return att
    if n / len(job.passages) < min_support:
        att.status = "low_support"
        return att
    att.status, att.target_id = "merged", tid
    att.visible_from, att.first_chunk = first[tid].reading_order, first[tid].chunk_id
    att.basis = [b for b in basis[tid] if b][:3]
    att.chapters = [list(c) for c in sorted({(p.book_id, p.chapter_idx) for p in attributed[tid]})]
    return att


async def attribute_epithets(
    jobs: Sequence[EpithetJob],
    chat: ChatModel,
    prompt: Prompt,
    m2e: Mapping[str, str],
    mentions: Mapping[str, MentionLike],
    real_name_segments: set[str],
    *,
    min_support: float,
    concurrency: int,
    stats: Counter,
    forms_of: Mapping[str, set[str]] | None = None,
    generic_mode: str = "link",
) -> list[Attribution]:
    by_chapter: dict[tuple[str, int], set[str]] = defaultdict(set)
    for mid, eid in m2e.items():
        m = mentions[mid]
        by_chapter[(m.book_id, m.chapter_idx)].add(eid)
    todo = [j for j in jobs if j.passages and j.candidates]

    async def run(job: EpithetJob) -> EpithetOut:
        res = await chat.achat(render_job(job, prompt), schema=EpithetOut, prompt_version=prompt.version_id)
        return res.parsed  # type: ignore[return-value]

    results = {r.job.entity_id: r for r in await run_jobs(todo, run, concurrency=concurrency, label="kg_epithet")}
    stats["epithet_llm_calls"] = len(todo)
    out = []
    for j in jobs:
        r = results.get(j.entity_id)
        if j.passages and not j.candidates:
            att = Attribution(j.entity_id, j.name, None, "no_target", n_passages=len(j.passages))
        else:
            parsed = r.value if r is not None and r.ok else None
            att = decide(
                j, parsed, by_chapter, real_name_segments,
                min_support=min_support, forms_of=forms_of, generic_mode=generic_mode,
            )  # fmt: skip
        stats[f"epithet_{att.status}"] += 1
        out.append(att)
    return out


def apply_attributions(
    entities: list[dict[str, Any]],
    m2e: dict[str, str],
    mentions: Mapping[str, MentionLike],
    attributions: Sequence[Attribution],
    stats: Counter,
) -> tuple[list[dict[str, Any]], set[str]]:
    """把归属结果落到实体上（原地修改 m2e），返回 (新的实体列表, 要删掉的泛称实体 ID)。
    专属称呼：从 visible_from 起的提及改挂目标并记作别名；更早的提及留在称呼实体上。
    泛称：挂靠章里的提及改挂对应的人，不记别名；有证据指向不同的人时，称呼实体若还有剩余提及，整体删掉。
    称呼实体没有剩余提及时直接去掉。"""
    by_id = {e["entity_id"]: e for e in entities}
    mids_of: dict[str, list[str]] = defaultdict(list)
    for mid, eid in m2e.items():
        mids_of[eid].append(mid)
    touched: set[str] = set()
    generic_drop: set[str] = set()
    for att in attributions:
        if att.generic and att.epithet_id in by_id:
            for b, c, tid, _ in att.links:
                if tid not in by_id:
                    continue
                moved = [
                    m
                    for m in mids_of[att.epithet_id]
                    if (mentions[m].book_id, mentions[m].chapter_idx) == (b, c) and m2e[m] == att.epithet_id
                ]
                for m in moved:
                    m2e[m] = tid
                stats["epithet_generic_mentions_linked"] += len(moved)
                touched.add(tid)
            touched.add(att.epithet_id)
            if att.multi:
                generic_drop.add(att.epithet_id)
            continue
        if att.status != "merged" or att.target_id not in by_id or att.epithet_id not in by_id:
            continue
        src, tgt = by_id[att.epithet_id], by_id[att.target_id]
        chapters = {(b, c) for b, c in att.chapters}
        moved = [
            m
            for m in mids_of[att.epithet_id]
            if mentions[m].reading_order >= att.visible_from  # type: ignore[operator]
            and (att.all_chapters or (mentions[m].book_id, mentions[m].chapter_idx) in chapters)
        ]
        for m in moved:
            m2e[m] = att.target_id
        stats["epithet_mentions_moved"] += len(moved)
        names = {a["alias"]: a for a in tgt.get("aliases", [])}
        moved_ms = sorted((mentions[m] for m in moved), key=lambda m: (m.reading_order, m.chunk_id))
        for form in [src["name"], *(a["alias"] for a in src.get("aliases", []))]:
            if form == tgt["name"]:
                continue
            # 每个写法从它自己在改挂提及中第一次出现的位置起可见（不早于归属起点）；没出现过的写法不记作别名
            hit = next((m for m in moved_ms if form in normalize_name(m.name) or normalize_name(m.name) in form), None)
            if hit is None:
                stats["epithet_alias_form_skipped"] += 1
                continue
            vis = max(att.visible_from, hit.reading_order)  # type: ignore[type-var]
            first = att.first_chunk if vis == att.visible_from else hit.chunk_id
            new = {"alias": form, "visible_from": vis, "first_chunk": first, "link": "epithet"}
            old = names.get(form)
            if old is None:
                tgt.setdefault("aliases", []).append(new)
                names[form] = new
            elif old["visible_from"] > vis:
                old.update(new)
        tgt["aliases"].sort(key=lambda a: (a["visible_from"], a["alias"]))
        touched.update({att.epithet_id, att.target_id})
    # 重算受影响实体的计数与首末位置
    mids_of = defaultdict(list)
    for mid, eid in m2e.items():
        mids_of[eid].append(mid)
    out = []
    for e in entities:
        eid = e["entity_id"]
        if eid in touched:
            ms = [mentions[m] for m in mids_of.get(eid, [])]
            if not ms:
                stats["epithet_entities_removed"] += 1
                generic_drop.discard(eid)
                continue
            if eid in generic_drop:
                stats["epithet_generic_entities_dropped"] += 1
                stats["epithet_generic_mentions_dropped"] += len(ms)
            e["n_mentions"] = sum(m.count for m in ms)
            e["n_chunks"] = len({m.chunk_id for m in ms})
            e["first_reading_order"] = min(m.reading_order for m in ms)
            e["last_reading_order"] = max(m.reading_order for m in ms)
            e["first_chunk"] = min(ms, key=lambda m: (m.reading_order, m.chunk_id)).chunk_id
        out.append(e)
    return out, generic_drop
