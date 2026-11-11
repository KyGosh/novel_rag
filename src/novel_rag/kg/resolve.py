"""实体消歧：把逐 chunk 的实体提及合并成系列级实体。

流程（P3 方案第六节）：
1. 规则归一：统一间隔号、去称谓（抽取阶段已算出 key）；组织名去掉"家族 / 家"后缀。
2. 初始聚类：按"类型 + 规范名"分组。
3. 拆分检查：同一全名下提及较多的人物簇交给 LLM，按描述判断是否同名不同人（辈分、年代、身份矛盾才拆）。
4. 简称链接：只有名字一段的人物提及（"琼恩"），逐个提及链接到含该片段的全名实体：同 chunk → 同章 →
   全书主导（只看首次出现不晚于本提及的候选），都不唯一时保留为独立实体。
5. 别名陈述（"人称小恶魔"）：原文有证据，直接合并。
6. 候选对：抽取时不被采信的"声称全名"、名称 + 描述向量的近邻、组织名包含关系。规则能定的直接合并，
   其余交给 LLM 判定 same / different / unsure；两个互不相干的全名永不合并（cannot-link）。
7. 可选的人工覆盖文件（data/books/{series}/aliases.toml）：指定合并或补别名。
8. 实体 ID = {series}:{type}:{规范名}，同名不同人加确定性限定语；别名带"首次可见位置"：与规范名没有共同片段的
   别名（绰号、化名），要等原文把它和这个人联系起来（别名陈述、同段共现、原文支持的全名）之后才可见。

消歧描述只在这里内部使用，不写进导出的实体，也不进 Neo4j（可能含后续剧情）。
"""

from __future__ import annotations

import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from novel_rag.batch import run_jobs
from novel_rag.config import KGConfig
from novel_rag.kg.names import NameRules, normalize_name, segments
from novel_rag.kg.schema import ENTITY_LABELS, ORG, PERSON, TITLE
from novel_rag.llm.base import ChatModel, Embedder
from novel_rag.prompts import Prompt

RESOLVE_VERSION = "kg-resolve-10"  # 10：消歧之后增加人物类别判定与关系清洗（P3.1）


def squash(s: str) -> str:
    return normalize_name(s).replace("·", "")


def group_key(g: str | None) -> str | None:
    if not g:
        return None
    n = normalize_name(g)
    for suf in ("家族", "家"):
        if n.endswith(suf) and len(n) - len(suf) >= 2:
            return n[: -len(suf)]
    return n or None


@dataclass
class Mention:
    mid: str
    chunk_id: str
    book_id: str
    chapter_idx: int
    reading_order: int
    type: str
    name: str
    full_name: str | None
    claimed: str | None
    key: str
    group: str | None
    description: str
    category: str | None
    count: int

    @property
    def display(self) -> str:
        return self.full_name or self.name


@dataclass
class Cluster:
    cid: int
    type: str
    key: str
    mids: list[str]
    qualifier: str | None = None
    split_family: int | None = None  # 由同一次拆分产生的兄弟簇共享这个编号（cannot-link）


class SplitPerson(BaseModel):
    qualifier: str = ""
    items: list[int] = Field(default_factory=list)


class SplitOut(BaseModel):
    persons: list[SplitPerson] = Field(default_factory=list)


class JudgeOut(BaseModel):
    verdict: str = Field(description="same / different / unsure")
    reason: str = ""


@dataclass
class Overrides:
    """aliases.toml：[[merge]] names = ["甲", "乙"]（同类型、按规范名找簇）；[[alias]] name = "甲" aliases = [...]；
    [[category]] name = "甲"（规范名或实体 ID）category = "人" / "动物" / "非角色"（纠正人物类别判定）。"""

    merges: list[list[str]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Overrides:
        if not path.is_file():
            return cls()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        merges = [list(x["names"]) for x in raw.get("merge", [])]
        aliases = {x["name"]: list(x["aliases"]) for x in raw.get("alias", [])}
        categories = {x["name"]: str(x["category"]) for x in raw.get("category", [])}
        return cls(merges, aliases, categories)


class Resolver:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        rules: NameRules,
        cfg: KGConfig,
        series_id: str,
        snippet: Callable[[Mention], str],
    ):
        self.rules, self.cfg, self.series_id, self.snippet = rules, cfg, series_id, snippet
        self.mentions: dict[str, Mention] = {}
        self.alias_statements: list[tuple[str, str, int, str]] = []  # (mid, alias, reading_order, chunk_id)
        for rec in records:
            for m in rec["mentions"]:
                mid = f"{rec['chunk_id']}#{m['m']}"
                self.mentions[mid] = Mention(
                    mid, rec["chunk_id"], rec["book_id"], rec["chapter_idx"], rec["reading_order"], m["type"],
                    m["name"], m["full_name"], m.get("claimed_full_name"),
                    normalize_name(m["full_name"] or m["name"]) if m["type"] == TITLE
                    else rules.canonical(m["full_name"] or m["name"]),
                    m["group"], m["description"],
                    m.get("category"), m.get("count", 1),
                )  # fmt: skip
            for a in rec["aliases"]:
                self.alias_statements.append(
                    (f"{rec['chunk_id']}#{a['m']}", a["alias"], rec["reading_order"], rec["chunk_id"])
                )
        self.clusters: dict[int, Cluster] = {}
        self.m2c: dict[str, int] = {}
        self.parent: dict[int, int] = {}
        self.stats: Counter = Counter()
        self.decisions: list[dict[str, Any]] = []  # 合并 / 拆分决定（内部记录，供质量评估与难例清单）
        self._next = 0
        self._split_families = 0

    # ---- 基础操作 ----

    def _new_cluster(self, type_: str, key: str, mids: list[str], **kw: Any) -> Cluster:
        c = Cluster(self._next, type_, key, mids, **kw)
        self._next += 1
        self.clusters[c.cid] = c
        self.parent[c.cid] = c.cid
        for mid in mids:
            self.m2c[mid] = c.cid
        return c

    def find(self, cid: int) -> int:
        while self.parent[cid] != cid:
            self.parent[cid] = self.parent[self.parent[cid]]
            cid = self.parent[cid]
        return cid

    def full_names(self, cid: int) -> set[str]:
        """簇内带间隔号的全名（去称谓、去间隔号后的形式）。"""
        out = set()
        for mid in self.clusters[cid].mids:
            d = self.rules.canonical(self.mentions[mid].display)
            if "·" in d:
                out.add(d)
        return out

    def cannot_link(self, a: int, b: int) -> str | None:
        ca, cb = self.clusters[a], self.clusters[b]
        if ca.type != cb.type:
            return "type"
        if ca.split_family is not None and ca.split_family == cb.split_family:
            return "split_siblings"
        fa, fb = self.full_names(a), self.full_names(b)
        if fa and fb and not ({squash(x) for x in fa} & {squash(x) for x in fb}):
            sa = {s for x in fa for s in segments(x)}
            sb = {s for x in fb for s in segments(x)}
            if not (sa & sb):
                return "unrelated_full_names"
            firsts_a, firsts_b = {segments(x)[0] for x in fa}, {segments(x)[0] for x in fb}
            lasts_a, lasts_b = {segments(x)[-1] for x in fa}, {segments(x)[-1] for x in fb}
            # 名相同、姓不同：通常是不同的人（琼恩·雪诺 / 琼恩·艾林）；但已婚女性有娘家姓与夫家姓两种写法
            # （凯特琳·徒利 / 凯特琳·史塔克），所以所属群体有交集时不禁止，交给判定
            if firsts_a & firsts_b and not (lasts_a & lasts_b) and not (self.top_groups(a) & self.top_groups(b)):
                return "same_given_different_surname"
        return None

    def union(self, a: int, b: int, reason: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        why = self.cannot_link(ra, rb)
        if why is not None:
            self.stats[f"merge_blocked_{why}"] += 1
            return False
        # 并到提及多的一方，保留其 cid
        if len(self.clusters[ra].mids) < len(self.clusters[rb].mids):
            ra, rb = rb, ra
        info = {"into_name": self.top_name(ra), "into_n": len(self.clusters[ra].mids),
                "from_name": self.top_name(rb), "from_n": len(self.clusters[rb].mids)}  # fmt: skip
        self.parent[rb] = ra
        self.clusters[ra].mids.extend(self.clusters[rb].mids)
        for mid in self.clusters[rb].mids:
            self.m2c[mid] = ra
        if self.clusters[ra].qualifier is None:
            self.clusters[ra].qualifier = self.clusters[rb].qualifier
        del self.clusters[rb]
        self.stats[f"merged_{reason}"] += 1
        self.decisions.append({"op": "merge", "reason": reason, "into": ra, "from": rb, **info})
        return True

    def top_groups(self, cid: int, n: int = 3) -> set[str]:
        g = Counter(group_key(self.mentions[m].group) for m in self.clusters[cid].mids if self.mentions[m].group)
        return {x for x, _ in g.most_common(n)}

    def top_name(self, cid: int) -> str:
        names = Counter(self.canon(self.mentions[m]) for m in self.clusters[cid].mids)
        return names.most_common(1)[0][0] if names else ""

    def canon(self, m: Mention) -> str:
        """提及的规范写法：去称谓（头衔除外）。"""
        return normalize_name(m.display) if m.type == TITLE else self.rules.canonical(m.display)

    def cluster_of(self, mid: str) -> int:
        return self.find(self.m2c[mid])

    # ---- 1–2 初始聚类 ----

    def mention_key(self, m: Mention) -> tuple[str, str]:
        k = group_key(m.key) if m.type == ORG else m.key
        return m.type, squash(k or m.key)

    def build_initial(self) -> None:
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for mid, m in self.mentions.items():
            groups[self.mention_key(m)].append(mid)
        for (t, k), mids in sorted(groups.items()):
            self._new_cluster(t, k, mids)
        self.stats["initial_clusters"] = len(self.clusters)

    def _has_full(self, c: Cluster) -> bool:
        return any("·" in self.mentions[mid].display for mid in c.mids)

    # ---- 3 拆分检查 ----

    def _desc_items(self, c: Cluster) -> list[tuple[str, int, str | None, list[str]]]:
        """（描述, 首次出现的 reading_order, 群体, 提及列表），按首次出现排序。"""
        by: dict[str, list[str]] = defaultdict(list)
        for mid in c.mids:
            by[self.mentions[mid].description.strip()].append(mid)
        items = []
        for d, mids in by.items():
            if not d:
                continue
            first = min(self.mentions[x].reading_order for x in mids)
            grp = Counter(group_key(self.mentions[x].group) for x in mids if self.mentions[x].group).most_common(1)
            items.append((d, first, grp[0][0] if grp else None, mids))
        items.sort(key=lambda x: (x[1], x[0]))
        return items

    async def split_check(self, chat: ChatModel, prompt: Prompt, embedder: Embedder) -> None:
        targets = [
            c
            for c in self.clusters.values()
            if c.type == PERSON and len(c.mids) >= self.cfg.split_min_mentions and self._has_full(c)
        ]
        self.stats["split_checked"] = len(targets)
        n_max = self.cfg.split_max_descriptions

        async def check(c: Cluster):
            items = self._desc_items(c)
            if len(items) < 2:
                return c, items, [], None
            if len(items) > n_max:
                idx = np.linspace(0, len(items) - 1, n_max).round().astype(int)
                sampled = [items[i] for i in sorted(set(idx.tolist()))]
            else:
                sampled = items
            lines = [f"{i}. [{ro}] {g or '-'} | {d}" for i, (d, ro, g, _) in enumerate(sampled, 1)]
            name = Counter(self.rules.canonical(self.mentions[m].display) for m in c.mids).most_common(1)[0][0]
            msgs = prompt.render(name=name, type=ENTITY_LABELS[c.type], items="\n".join(lines))
            res = await chat.achat(msgs, schema=SplitOut, prompt_version=prompt.version_id)
            return c, items, sampled, res.parsed

        results = await run_jobs(targets, check, concurrency=self.cfg.jobs_in_flight, label="kg_split")
        for r in results:
            if not r.ok:
                self.stats["split_failed"] += 1
                continue
            c, items, sampled, out = r.value
            if out is None or len(out.persons) <= 1:
                continue
            self._apply_split(c, items, sampled, out, embedder)

    def _apply_split(self, c: Cluster, items, sampled, out: SplitOut, embedder: Embedder) -> None:
        n = len(sampled)
        assign: dict[int, int] = {}  # 采样描述下标 → 组号
        quals = []
        for g, p in enumerate(out.persons):
            quals.append(normalize_name(p.qualifier)[:8] or f"其{g + 1}")
            for i in p.items:
                if 1 <= i <= n and (i - 1) not in assign:
                    assign[i - 1] = g
        if len(set(assign.values())) <= 1:
            return
        texts = [d for d, *_ in items]
        vecs = embedder.embed_documents(texts)
        sidx = {d: i for i, (d, *_) in enumerate(items)}
        groups = sorted(set(assign.values()))
        cent = {g: np.mean([vecs[sidx[sampled[i][0]]] for i, gg in assign.items() if gg == g], axis=0) for g in groups}
        desc_group: dict[str, int] = {}
        for d, *_ in items:
            si = next((i for i, s in enumerate(sampled) if s[0] == d), None)
            if si is not None and si in assign:
                desc_group[d] = assign[si]
            else:
                v = vecs[sidx[d]]
                desc_group[d] = max(groups, key=lambda g: float(v @ cent[g]))
        members: dict[int, list[str]] = defaultdict(list)
        for mid in c.mids:
            d = self.mentions[mid].description.strip()
            members[desc_group.get(d, -1)].append(mid)
        # 没有描述的提及：归入同章提及最多的组，否则最大的组
        orphan = members.pop(-1, [])
        sizes = {g: len(v) for g, v in members.items()}
        for mid in orphan:
            ch = (self.mentions[mid].book_id, self.mentions[mid].chapter_idx)
            near = Counter(
                g
                for g, v in members.items()
                for x in v
                if (self.mentions[x].book_id, self.mentions[x].chapter_idx) == ch
            )
            g = near.most_common(1)[0][0] if near else max(sizes, key=lambda k: sizes[k])
            members[g].append(mid)
        # 提及太少的组并回最大组（防止过度拆分）
        big = max(members, key=lambda g: len(members[g]))
        for g in list(members):
            if g != big and len(members[g]) < 2:
                members[big].extend(members.pop(g))
        if len(members) <= 1:
            self.stats["split_rejected_small"] += 1
            return
        fam = self._split_families
        self._split_families += 1
        c.mids = members[big]
        c.qualifier = quals[big] if big < len(quals) else None
        c.split_family = fam
        for g, mids in sorted(members.items()):
            if g == big:
                continue
            nc = self._new_cluster(c.type, c.key, mids, qualifier=quals[g] if g < len(quals) else None)
            nc.split_family = fam
        for mid in c.mids:
            self.m2c[mid] = c.cid
        self.stats["split_applied"] += 1
        self.stats["split_new_entities"] += len(members) - 1
        self.decisions.append(
            {
                "op": "split",
                "key": c.key,
                "groups": {quals[g] if g < len(quals) else str(g): len(v) for g, v in members.items()},
            }
        )

    # ---- 4 简称链接 ----

    def link_short_names(self) -> None:
        full = [c for c in self.clusters.values() if c.type == PERSON and self._has_full(c)]
        first_idx: dict[str, list[int]] = defaultdict(list)
        chunks: dict[int, set[str]] = {}
        chapters: dict[int, Counter] = {}
        total: dict[int, int] = {}
        groups: dict[int, set[str]] = {}
        for c in full:
            segs = Counter()
            for mid in c.mids:
                d = self.rules.canonical(self.mentions[mid].display)
                if "·" in d:
                    s = segments(d)
                    segs[(s[0], s[-1])] += 1
            for (a, _b), _ in segs.most_common(2):
                if c.cid not in first_idx[a]:
                    first_idx[a].append(c.cid)
            chunks[c.cid] = {self.mentions[m].chunk_id for m in c.mids}
            chapters[c.cid] = Counter((self.mentions[m].book_id, self.mentions[m].chapter_idx) for m in c.mids)
            total[c.cid] = len(c.mids)
            groups[c.cid] = {group_key(self.mentions[m].group) for m in c.mids if self.mentions[m].group}

        def by_group(m: Mention, cands: list[int]) -> int | None:
            g = group_key(m.group)
            hit = [c for c in cands if g and g in groups[c]]
            return hit[0] if len(hit) == 1 else None

        def pick(m: Mention, cands: list[int], dominance: bool) -> tuple[int | None, str]:
            # 本段里另列为一个实体的候选，就不是这个简称所指的人（抽取要求同一实体在本段只列一次）
            cands = [c for c in cands if m.chunk_id not in chunks[c]]
            if not cands:
                return None, "unresolved"
            ch = (m.book_id, m.chapter_idx)
            in_ch = sorted(((chapters[c][ch], c) for c in cands if chapters[c][ch] > 0), reverse=True)
            if len(in_ch) == 1:
                return in_ch[0][1], "chapter"
            if len(in_ch) > 1:
                if in_ch[0][0] >= 2 * in_ch[1][0]:
                    return in_ch[0][1], "chapter_dominant"
                return by_group(m, [c for _, c in in_ch]), "chapter_group"
            if dominance:
                # 全书范围：唯一候选，或提及数是第二名的 4 倍以上（链接本身不暴露名字，别名可见性另行控制）
                ranked = sorted(((total[c], c) for c in cands), reverse=True)
                if ranked and (len(ranked) == 1 or ranked[0][0] >= 4 * ranked[1][0]):
                    return ranked[0][1], "global_dominant"
            return None, "unresolved"

        full_count = {c.cid: sum("·" in self.mentions[m].display for m in c.mids) for c in full}
        shorts = [c for c in list(self.clusters.values()) if c.type == PERSON and not self._has_full(c)]
        for c in shorts:
            key = c.key
            fc = first_idx.get(key, [])
            if not fc:
                continue
            # 全书主导只在候选本身有一定分量时成立：几百个"丹妮"不能因为全书只有一个"丹妮·某某"就全挂到他名下
            dominance_ok = max(full_count[x] for x in fc) >= self.cfg.short_link_min_ratio * len(c.mids)
            keep = []
            for mid in c.mids:
                m = self.mentions[mid]
                target, how = pick(m, fc, dominance_ok) if fc else (None, "unresolved")
                if target is None:
                    keep.append(mid)
                    self.stats["short_unresolved"] += 1
                    continue
                self.clusters[target].mids.append(mid)
                self.m2c[mid] = target
                self.stats[f"short_linked_{how}"] += 1
            if keep:
                c.mids = keep
            else:
                del self.clusters[c.cid]
                self.parent[c.cid] = c.cid

    def drop_surname_clusters(self) -> None:
        """只有姓、与某个家族同名的人物簇（"史塔克大人""兰尼斯特"）指代不明：在任何合并之前移除，
        否则它会成为把不同的人串在一起的枢纽。提及不映射到实体。"""
        org_keys = {c.key for c in self.clusters.values() if c.type == ORG}
        for c in list(self.clusters.values()):
            if c.type == PERSON and "·" not in c.key and c.key in org_keys and not self._has_full(c):
                self.stats["dropped_surname_only_mentions"] += len(c.mids)
                for mid in c.mids:
                    self.m2c.pop(mid, None)
                del self.clusters[c.cid]

    # ---- 5 别名陈述 ----

    def _key_index(self) -> dict[tuple[str, str], list[int]]:
        idx: dict[tuple[str, str], list[int]] = defaultdict(list)
        for c in self.clusters.values():
            idx[(c.type, c.key)].append(c.cid)
        return idx

    def apply_alias_statements(self) -> None:
        """别名陈述（"人称小恶魔"）→ 把以该别名为名的簇并入陈述的主体。三道保护，防止错误陈述把不同的人串在一起：
        1. 真名保护：别名是另一个全名人物的名或姓（"凯特琳""詹姆""莫尔蒙"）时不合并——那是真名，不是绰号；
        2. 歧义：多个实体都声称这个别名时，只有支持数最高者达到第二名的 3 倍才合并；
        3. 支持度：陈述次数至少是目标簇提及数的 alias_merge_min_support（一两句错误陈述吞不掉一个大簇）。
        不满足条件的别名仍记作别名（finalize 时按陈述出现位置计可见性），只是不合并簇。"""
        idx = self._key_index()
        support: dict[tuple[str, str], Counter] = defaultdict(Counter)  # (类型, 别名键) → {主体簇: 陈述数}
        for mid, alias, _ro, _chunk in self.alias_statements:
            if mid not in self.m2c:
                continue
            e = self.cluster_of(mid)
            t = self.clusters[e].type
            k = squash(group_key(alias) if t == ORG else self.rules.canonical(alias))
            if k:
                support[(t, k)][e] += 1
        # 每个全名人物簇的名字片段（真名保护用）
        seg_owner: dict[str, set[int]] = defaultdict(set)
        for c in self.clusters.values():
            if c.type != PERSON:
                continue
            for fn in self.full_names(c.cid):
                for sgm in segments(fn):
                    seg_owner[squash(sgm)].add(c.cid)
                seg_owner[squash(fn)].add(c.cid)
        for (t, k), srcs in sorted(support.items()):
            targets = sorted({self.find(x) for x in idx.get((t, k), []) if self.find(x) in self.clusters})
            if not targets:
                continue
            if len(targets) > 1:
                self.stats["alias_target_ambiguous"] += 1
                continue
            target = targets[0]
            merged: Counter = Counter()
            for src, n in srcs.items():
                merged[self.find(src)] += n
            ranked = merged.most_common()
            src, n = ranked[0]
            if src == target:
                continue
            if len(ranked) > 1 and n < 3 * ranked[1][1]:
                self.stats["alias_blocked_contested"] += 1
                continue
            owners = {self.find(o) for o in seg_owner.get(k, set())} - {src}
            if t == PERSON and owners:  # 别名是别人全名里的名或姓：是真名，不是绰号
                self.stats["alias_blocked_real_name"] += 1
                continue
            if n < max(1.0, self.cfg.alias_merge_min_support * len(self.clusters[target].mids)):
                self.stats["alias_blocked_low_support"] += 1
                continue
            self.union(src, target, "alias_statement")

    def apply_overrides(self, ov: Overrides) -> None:
        idx = self._key_index()
        for names in ov.merges:
            cids = []
            for n in names:
                hits = [c for t in (PERSON, ORG) for c in idx.get((t, squash(self.rules.canonical(n))), [])]
                if len(hits) == 1:
                    cids.append(hits[0])
                else:
                    self.stats["override_unresolved"] += 1
            for other in cids[1:]:
                self.union(cids[0], other, "override")

    # ---- 6 候选对与判定 ----

    def profile(self, cid: int, *, n_desc: int = 3) -> str:
        c = self.clusters[cid]
        names = Counter(self.rules.canonical(self.mentions[m].display) for m in c.mids)
        descs = Counter(self.mentions[m].description for m in c.mids if self.mentions[m].description)
        groups = Counter(group_key(self.mentions[m].group) for m in c.mids if self.mentions[m].group)
        text = f"{names.most_common(1)[0][0]}（{ENTITY_LABELS[c.type]}）：" + "；".join(
            d for d, _ in descs.most_common(n_desc)
        )
        if groups:
            text += f"。群体：{groups.most_common(1)[0][0]}"
        return text

    def judge_block(self, cid: int) -> str:
        c = self.clusters[cid]
        ms = sorted((self.mentions[m] for m in c.mids), key=lambda m: (m.reading_order, m.mid))
        names = Counter(self.rules.canonical(m.display) for m in ms)
        groups = Counter(group_key(m.group) for m in ms if m.group)
        descs: list[str] = []
        for m in ms:
            if m.description and m.description not in descs:
                descs.append(m.description)
        step = max(1, len(descs) // 6)
        descs = descs[::step][:6]
        quotes = []
        seen_chunks = set()
        for m in ms[:: max(1, len(ms) // max(1, self.cfg.judge_evidence_quotes))]:
            if m.chunk_id in seen_chunks:
                continue
            seen_chunks.add(m.chunk_id)
            q = self.snippet(m)
            if q:
                quotes.append(q)
            if len(quotes) >= self.cfg.judge_evidence_quotes:
                break
        lines = [
            f"- 类型：{ENTITY_LABELS[c.type]}；提及 {len(ms)} 次",
            "- 名字写法：" + "、".join(f"{n}（{k}）" for n, k in names.most_common(5)),
            "- 所属群体：" + ("、".join(g for g, _ in groups.most_common(3)) or "（无）"),
            "- 描述：" + "；".join(descs),
            "- 原文片段：" + " / ".join(f"「{q}」" for q in quotes),
        ]
        return "\n".join(lines)

    def candidate_pairs(self, embedder: Embedder) -> list[tuple[float, int, int, str]]:
        """(优先级分数, a, b, 来源)。分数越大越先判定。"""
        pairs: dict[tuple[int, int], tuple[float, str]] = {}

        def add(a: int, b: int, score: float, src: str) -> None:
            a, b = self.find(a), self.find(b)
            if a == b:
                return
            k = (min(a, b), max(a, b))
            if k not in pairs or pairs[k][0] < score:
                pairs[k] = (score, src)

        idx = self._key_index()
        # 声称全名：抽取时模型给出、但原文不支持的全名
        for c in list(self.clusters.values()):
            claims = Counter(self.mentions[m].claimed for m in c.mids if self.mentions[m].claimed)
            for claimed, n in claims.items():
                k = (c.type, squash(self.rules.canonical(claimed)))
                hits = [x for x in idx.get(k, []) if x in self.clusters and x != c.cid]
                if len(hits) == 1:
                    add(c.cid, hits[0], 2.0 + min(n, 10) / 10, "claimed")
        # 组织名包含关系（"临冬城史塔克" ⊃ "史塔克"）
        orgs = [c for c in self.clusters.values() if c.type == ORG and len(c.key) >= 2]
        by_key = {c.key: c.cid for c in orgs}
        for c in orgs:
            for L in range(2, len(c.key)):
                for i in range(0, len(c.key) - L + 1):
                    sub = c.key[i : i + L]
                    if sub in by_key and by_key[sub] != c.cid and (i == 0 or i + L == len(c.key)):
                        add(c.cid, by_key[sub], 1.5, "org_containment")
        # 人名前缀：只有一段的名字是另一个实体名第一段的前缀（"丹妮" / "丹妮莉丝"）
        persons = [c for c in self.clusters.values() if c.type == PERSON]
        first_seg: dict[str, list[int]] = defaultdict(list)
        for c in persons:
            names = {segments(self.rules.canonical(self.mentions[m].display))[0] for m in c.mids}
            for n in names:
                first_seg[n].append(c.cid)
        for c in persons:
            if "·" in c.key or len(c.key) < 2:
                continue
            for n, cids in first_seg.items():
                if len(n) > len(c.key) and n.startswith(c.key):
                    for other in cids:
                        if other != c.cid:
                            add(c.cid, other, 1.2, "name_prefix")
        # 名称 + 描述向量近邻
        for t in sorted({c.type for c in self.clusters.values()}):
            cs = sorted((c for c in self.clusters.values() if c.type == t), key=lambda c: c.cid)
            if len(cs) < 2:
                continue
            vecs = embedder.embed_documents([self.profile(c.cid) for c in cs]).astype(np.float32)
            k = min(self.cfg.candidate_top_k + 1, len(cs))
            for start in range(0, len(cs), 2048):
                sims = vecs[start : start + 2048] @ vecs.T
                top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                for r, row in enumerate(top):
                    i = start + r
                    for j in row:
                        if j == i or sims[r, j] < self.cfg.candidate_min_sim:
                            continue
                        add(cs[i].cid, cs[j].cid, float(sims[r, j]), "embedding")
        out = []
        for (a, b), (score, src) in pairs.items():
            why = self.cannot_link(a, b)
            if why is not None:
                self.stats[f"candidate_blocked_{why}"] += 1
                continue
            out.append((score, a, b, src))
        # 先判"两边都有多次提及"的对，再按分数
        out.sort(
            key=lambda x: (-(min(len(self.clusters[x[1]].mids), len(self.clusters[x[2]].mids)) >= 2), -x[0], x[1], x[2])
        )
        self.stats["candidate_pairs"] = len(out)
        return out

    def _names_overlap(self, a: int, b: int) -> bool:
        sa = {s for m in self.clusters[a].mids for s in segments(self.rules.canonical(self.mentions[m].display))}
        sb = {s for m in self.clusters[b].mids for s in segments(self.rules.canonical(self.mentions[m].display))}
        return bool(sa & sb)

    def _groups_conflict(self, a: int, b: int) -> bool:
        def top(c: int) -> str | None:
            g = Counter(group_key(self.mentions[m].group) for m in self.clusters[c].mids if self.mentions[m].group)
            return g.most_common(1)[0][0] if g else None

        ga, gb = top(a), top(b)
        return bool(ga and gb and ga != gb)

    async def judge_pairs(self, pairs: list[tuple[float, int, int, str]], chat: ChatModel, prompt: Prompt) -> None:
        to_judge = []
        for score, a, b, src in pairs:
            a, b = self.find(a), self.find(b)  # 之前的规则合并可能已经把某一方并走了
            if a == b:
                continue
            if (
                src == "embedding"
                and score >= self.cfg.auto_merge_sim
                and self._names_overlap(a, b)
                and not self._groups_conflict(a, b)
            ):
                self.union(a, b, "rule_auto")
                continue
            to_judge.append((score, a, b, src))
        # 规则合并都做完后再归一一次：去掉已合并的、重复的对，并重新检查 cannot-link
        normalized, seen = [], set()
        for score, a, b, src in to_judge:
            a, b = self.find(a), self.find(b)
            key = (min(a, b), max(a, b))
            if a == b or key in seen or self.cannot_link(a, b) is not None:
                continue
            seen.add(key)
            normalized.append((score, a, b, src))
        to_judge = normalized[: self.cfg.judge_max_pairs]
        self.stats["judge_pairs"] = len(to_judge)
        blocks = {c: None for _, a, b, _ in to_judge for c in (a, b)}
        for c in blocks:
            blocks[c] = self.judge_block(c)

        async def judge(p):
            _, a, b, _ = p
            res = await chat.achat(
                prompt.render(a=blocks[a], b=blocks[b]), schema=JudgeOut, prompt_version=prompt.version_id
            )
            return res.parsed

        results = await run_jobs(to_judge, judge, concurrency=self.cfg.jobs_in_flight, label="kg_judge")
        for r in results:
            score, a, b, src = r.job
            if not r.ok:
                self.stats["judge_failed"] += 1
                continue
            v = (r.value.verdict or "").strip().lower()
            self.stats[f"judge_{v if v in ('same', 'different', 'unsure') else 'invalid'}"] += 1
            if v == "same":
                self.union(a, b, f"judge_{src}")

    # ---- 8 定稿：实体、别名、提及映射 ----

    def finalize(self) -> tuple[list[dict[str, Any]], dict[str, str]]:
        ents = []
        for c in self.clusters.values():
            ms = [self.mentions[m] for m in c.mids]
            if not ms:
                continue
            names = Counter(self.canon(m) for m in ms)
            first_of: dict[str, int] = {}
            for m in sorted(ms, key=lambda m: m.reading_order):
                first_of.setdefault(self.canon(m), m.reading_order)
            full = [n for n in names if "·" in n]
            pool = full or list(names)
            canonical = sorted(pool, key=lambda n: (-names[n], first_of[n], n))[0]
            ents.append((c, ms, canonical, first_of[canonical]))
        ents.sort(key=lambda x: (min(m.reading_order for m in x[1]), x[2], -len(x[1]), x[0].cid))
        key_owner: dict[str, int] = {}  # 规范名键 → 以它为规范名的簇（别名陈述指向别的实体时不记作本实体的别名）
        for c, _ms, canonical, _nv in ents:
            key_owner.setdefault(squash(canonical), c.cid)
        used: Counter = Counter()
        out, m2e = [], {}
        for c, ms, canonical, name_visible in ents:
            base = f"{self.series_id}:{c.type.lower()}:{canonical}"
            eid = base
            if used[base]:
                q = c.qualifier
                if not q:
                    g = Counter(group_key(m.group) for m in ms if m.group).most_common(1)
                    q = g[0][0] if g else str(used[base] + 1)
                eid = f"{base}|{q}"
                while eid in used:
                    eid += "'"
            used[base] += 1
            used[eid] += 1
            for m in ms:
                m2e[m.mid] = eid
            cats = Counter(m.category for m in ms if m.category)
            ents_rec = {
                "entity_id": eid,
                "type": c.type,
                "name": canonical,
                "name_visible_from": name_visible,  # 规范名首次出现的位置；之前只能用可见的别名称呼
                "qualifier": c.qualifier if eid != base else None,
                "category": cats.most_common(1)[0][0] if cats else None,
                "groups": [g for g, _ in Counter(group_key(m.group) for m in ms if m.group).most_common(3)],
                "n_mentions": sum(m.count for m in ms),
                "n_chunks": len({m.chunk_id for m in ms}),
                "first_reading_order": min(m.reading_order for m in ms),
                "last_reading_order": max(m.reading_order for m in ms),
                "first_chunk": min(ms, key=lambda m: (m.reading_order, m.chunk_id)).chunk_id,
                "aliases": self._aliases(canonical, ms, {k for k, cid in key_owner.items() if cid != c.cid}),
            }
            out.append(ents_rec)
        self.stats["entities"] = len(out)
        return out, m2e

    def _aliases(self, canonical: str, ms: list[Mention], others: set[str] = frozenset()) -> list[dict[str, Any]]:
        """别名及其首次可见位置。与规范名有共同片段的写法，首次出现即可见；没有共同片段的（绰号、化名），
        要等原文把它和这个人联系起来：别名陈述、与有共同片段的写法同段出现，或原文支持的全名。"""
        canon_segs = set(segments(canonical))
        by_chunk: dict[str, set[str]] = defaultdict(set)
        first: dict[str, tuple[int, str]] = {}
        grounded: dict[str, tuple[int, str, str]] = {}
        for m in sorted(ms, key=lambda m: (m.reading_order, m.chunk_id)):
            for v in {self.rules.canonical(m.name), self.rules.canonical(m.display)}:
                if v and v != canonical:
                    first.setdefault(v, (m.reading_order, m.chunk_id))
            by_chunk[m.chunk_id].add(self.rules.canonical(m.name))
            if m.full_name:  # 原文支持的全名：本段把简称和全名联系起来了
                grounded.setdefault(self.rules.canonical(m.name), (m.reading_order, m.chunk_id, "full_name"))
        for chunk, vs in by_chunk.items():
            if any(set(segments(v)) & canon_segs for v in vs):
                ro = next(m.reading_order for m in ms if m.chunk_id == chunk)
                for v in vs:
                    if v not in grounded or grounded[v][0] > ro:
                        grounded[v] = (ro, chunk, "cooccur")
        mids = {m.mid for m in ms}
        for mid, alias, ro, chunk in self.alias_statements:
            if mid in mids:
                a = self.rules.canonical(alias)
                if squash(a) in others:  # 这个名字是另一个实体的规范名：陈述多半有误，不记作别名
                    self.stats["alias_statement_points_elsewhere"] += 1
                    continue
                if a != canonical:
                    first.setdefault(a, (ro, chunk))
                    if a not in grounded or grounded[a][0] > ro:
                        grounded[a] = (ro, chunk, "alias_statement")
        out = []
        for v, (ro, chunk) in sorted(first.items(), key=lambda x: (x[1][0], x[0])):
            if self.rules.is_generic(v):
                continue
            vs = set(segments(v))
            shares = bool(vs & canon_segs) or v in canonical or canonical in v
            shares = shares or any(
                s.startswith(c) or c.startswith(s) for s in vs for c in canon_segs if min(len(s), len(c)) >= 2
            )
            if shares:
                out.append({"alias": v, "visible_from": ro, "first_chunk": chunk, "link": "name_overlap"})
            elif v in grounded:
                gro, gchunk, how = grounded[v]
                out.append({"alias": v, "visible_from": max(ro, gro), "first_chunk": gchunk, "link": how})
            else:
                out.append({"alias": v, "visible_from": ro, "first_chunk": chunk, "link": "ungrounded"})
        return out
