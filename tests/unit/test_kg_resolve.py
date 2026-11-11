"""实体消歧与图谱组装：初始聚类、简称链接、别名陈述合并、cannot-link、候选判定（Fake LLM）、
别名可见位置（防剧透）、关系聚合不覆盖、事件参与者映射。样例全部自拟。"""

import asyncio
import json
from collections import Counter

from novel_rag.config import KGConfig
from novel_rag.kg.build import ChapterNameIndex, aggregate_relations, build_events
from novel_rag.kg.names import NameRules
from novel_rag.kg.resolve import Resolver
from novel_rag.llm.fake import FakeChatModel, FakeEmbedder
from novel_rag.prompts import load_prompt

RULES = NameRules()
CFG = KGConfig(split_min_mentions=100, candidate_min_sim=2.0)  # 单测里不做拆分，向量候选关掉


def M(m, name, type_="Person", full=None, group=None, desc="", claimed=None):
    return {
        "m": m, "name": name, "full_name": full, "claimed_full_name": claimed, "key": RULES.canonical(full or name),
        "type": type_, "category": "家族" if type_ == "Organization" else None, "group": group,
        "description": desc, "count": 1,
    }  # fmt: skip


def R(s, t, o, role=None, ev=(0, 4)):
    return {"s": s, "type": t, "role": role, "o": o, "evidence": list(ev), "match": "exact"}


def rec(chunk, ro, ch, mentions, relations=(), aliases=()):
    return {
        "chunk_id": chunk, "book_id": "demo-1", "chapter_idx": ch, "reading_order": ro, "section_type": "narrative",
        "mentions": mentions, "relations": list(relations), "aliases": list(aliases),
    }  # fmt: skip


RECORDS = [
    rec(
        "demo-1:0001:0000", 1, 1,
        [M(0, "阿岚·铁杉", group="铁杉家族", desc="灰岩堡城主"), M(1, "米拉", desc="阿岚的女儿"),
         M(2, "铁杉家族", "Organization"), M(3, "灰岩堡", "Location")],
        [R(0, "KIN", 1, "father"), R(0, "MEMBER_OF", 2)],
        [{"m": 0, "alias": "老灰狼", "evidence": [0, 4], "match": "exact"}],
    ),
    rec("demo-1:0001:0001", 1, 1, [M(0, "阿岚", desc="城主"), M(1, "米拉")], [R(0, "KIN", 1, "parent", ev=(10, 14))]),
    # 第 2 章：别名单独出现（此前已有别名陈述，可合并）；另一个同名不同姓的阿岚
    rec("demo-1:0002:0000", 2, 2, [M(0, "老灰狼", desc="灰发老人"), M(1, "阿岚·白桦", group="白桦家族")]),
    # 同段里另列了"阿岚·白桦"，所以这里的"阿岚"不是白桦；本章没有其他阿岚 → 全书主导（铁杉）
    rec("demo-1:0002:0001", 2, 2, [M(0, "阿岚"), M(1, "阿岚·白桦", group="白桦家族")], [R(0, "ALLY", 1)]),
    rec("demo-1:0003:0000", 3, 3, [M(0, "铁杉", desc="某位铁杉"), M(1, "米拉·铁杉", desc="阿岚之女")]),
    rec(
        "demo-1:0003:0001", 3, 3,
        [M(0, "雪狐", desc="神秘的旅人", claimed="米拉·铁杉"), M(1, "铁杉家族", "Organization")],
    ),
]  # fmt: skip


def _resolve(judge_verdict="same"):
    r = Resolver(RECORDS, RULES, CFG, "demo", snippet=lambda m: "")
    r.build_initial()
    r.link_short_names()
    r.drop_surname_clusters()
    r.apply_alias_statements()
    chat = FakeChatModel(lambda msgs: json.dumps({"verdict": judge_verdict, "reason": ""}))
    pairs = r.candidate_pairs(FakeEmbedder())
    asyncio.run(r.judge_pairs(pairs, chat, load_prompt("kg_judge")))
    return r, pairs


def test_resolve_merges_and_blocks():
    r, pairs = _resolve()
    ents, m2e = r.finalize()
    by_name = {e["name"]: e for e in ents}
    alan = by_name["阿岚·铁杉"]
    # 简称"阿岚"（同章 / 全书主导）与别名"老灰狼"（别名陈述）都并到阿岚·铁杉
    assert m2e["demo-1:0001:0001#0"] == alan["entity_id"] == m2e["demo-1:0002:0000#0"] == m2e["demo-1:0002:0001#0"]
    # 同名不同姓永不合并
    assert by_name["阿岚·白桦"]["entity_id"] != alan["entity_id"]
    # "米拉" 与 "米拉·铁杉"：简称按名链接
    assert m2e["demo-1:0001:0000#1"] == m2e["demo-1:0003:0000#1"]
    # 只有姓、与家族同名的人物提及不建实体
    assert "demo-1:0003:0000#0" not in m2e and r.stats["dropped_surname_only_mentions"] == 1
    # 声称全名产生候选对，Fake 判定 same → 合并
    assert any(src == "claimed" for *_, src in pairs)
    assert m2e["demo-1:0003:0001#0"] == m2e["demo-1:0003:0000#1"]
    assert r.stats["merged_alias_statement"] == 1


def test_judge_unsure_does_not_merge():
    r, _ = _resolve("unsure")
    _, m2e = r.finalize()
    assert m2e["demo-1:0003:0001#0"] != m2e["demo-1:0003:0000#1"]


def test_alias_visibility_waits_for_grounding():
    r, _ = _resolve()
    ents, _ = r.finalize()
    mira = next(e for e in ents if e["name"] == "米拉·铁杉")
    al = {a["alias"]: a for a in mira["aliases"]}
    assert al["米拉"]["link"] == "name_overlap" and al["米拉"]["visible_from"] == 1
    # "雪狐"只靠判定合并、原文从没把它和米拉联系起来：标记为 ungrounded，供难例清单与人工覆盖
    assert al["雪狐"]["link"] == "ungrounded"
    alan = next(e for e in ents if e["name"] == "阿岚·铁杉")
    assert {a["alias"]: a["link"] for a in alan["aliases"]}["老灰狼"] == "alias_statement"
    assert alan["name_visible_from"] == 1 and mira["name_visible_from"] == 3


def test_entity_ids_are_deterministic():
    a, _ = _resolve()
    b, _ = _resolve()
    ea, ma = a.finalize()
    eb, mb = b.finalize()
    assert [e["entity_id"] for e in ea] == [e["entity_id"] for e in eb] and ma == mb
    assert all(e["entity_id"].startswith("demo:") for e in ea)


def test_aggregate_relations_never_overwrites():
    r, _ = _resolve()
    _, m2e = r.finalize()
    stats = Counter()
    rels = aggregate_relations(RECORDS, m2e, stats)
    kin = [x for x in rels if x["type"] == "KIN"]
    assert len(kin) == 1 and kin[0]["role"] == "father"  # "父母"并入已有的"父亲"
    assert kin[0]["n_evidence"] == 2 and kin[0]["first_reading_order"] == 1
    assert stats["kin_parent_merged"] == 1
    ally = [x for x in rels if x["type"] == "ALLY"]
    assert len(ally) == 1 and ally[0]["subject"] < ally[0]["object"]  # 对称关系按 ID 排序存一个方向
    # 同一对实体先后出现不同关系：都保留
    more = RECORDS + [rec("demo-1:0004:0000", 4, 4, [M(0, "阿岚"), M(1, "阿岚·白桦")], [R(0, "ENEMY", 1)])]
    r2 = Resolver(more, RULES, CFG, "demo", snippet=lambda m: "")
    r2.build_initial()
    r2.link_short_names()
    _, m2e2 = r2.finalize()
    types = {x["type"] for x in aggregate_relations(more, m2e2, Counter()) if x["type"] in ("ALLY", "ENEMY")}
    assert types == {"ALLY", "ENEMY"}


def test_events_map_participants():
    r, _ = _resolve()
    _, m2e = r.finalize()
    idx = ChapterNameIndex(RECORDS, m2e, RULES)
    summaries = [
        {
            "section_type": "narrative", "book_id": "demo-1", "chapter_idx": 1, "chapter_title": "米拉",
            "reading_order": 1,
            "events": [{"event_id": "demo-1:0001:e01", "description": "阿岚带女儿巡视城堡",
                        "participants": ["阿岚", "无名氏"], "location": "灰岩堡",
                        "chunk_ids": ["demo-1:0001:0000"]}],
        }
    ]  # fmt: skip
    stats = Counter()
    events, rels = build_events(summaries, idx, stats)
    assert events[0]["participants"] == [m2e["demo-1:0001:0000#0"]]
    assert events[0]["unmapped_participants"] == ["无名氏"] and stats["event_participants_unmapped"] == 1
    assert {x["type"] for x in rels} == {"PARTICIPATES", "OCCURS_AT"}
    assert all(x["first_reading_order"] == 1 for x in rels)
