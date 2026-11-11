"""P3.1：人物类别（人 / 动物 / 非角色）、关系按类别清洗、常用词名字的弱匹配。样例全部自拟。"""

import asyncio
import json
from collections import Counter

import pytest

from novel_rag.graph_retrieval.linking import EntityLinker
from novel_rag.graph_retrieval.render import render
from novel_rag.graph_retrieval.templates import QueryResult
from novel_rag.kg.category import (
    CorpusCounter,
    EntityCard,
    apply_category_overrides,
    assign_categories,
    batches,
    clean_relations,
    drop_entities,
    entity_cards,
    judge_categories,
    linked_form_counts,
    load_word_list,
    person_category_of,
    weak_names,
)
from novel_rag.kg.resolve import Overrides
from novel_rag.kg.schema import ANIMAL, HUMAN, NON_CHARACTER, Schema, category_allows
from novel_rag.llm.fake import FakeChatModel
from novel_rag.prompts import load_prompt
from novel_rag.retrieval.routes import GraphRoute

ALAN, MIRA, STONE, PEBBLE, COIN = (
    "demo:person:阿岚·铁杉", "demo:person:米拉·铁杉", "demo:person:石头", "demo:person:卵石", "demo:person:银鹿",
)  # fmt: skip
FAMILY, KEEP = "demo:organization:铁杉", "demo:location:灰岩堡"


def ent(eid, type_="Person", name=None, aliases=(), groups=(), n=10, category=None):
    return {
        "entity_id": eid, "type": type_, "name": name or eid.rsplit(":", 1)[1], "groups": list(groups),
        "n_mentions": n, "category": category, "first_reading_order": 1, "last_reading_order": 9,
        "aliases": [{"alias": a, "visible_from": 1, "first_chunk": "", "link": "x"} for a in aliases],
    }  # fmt: skip


def ev(ro, start=0, chunk=None):
    return {"chunk_id": chunk or f"demo-1:{ro:04d}:0000", "reading_order": ro, "start": start, "end": start + 4,
            "book_id": "demo-1"}  # fmt: skip


def rel(s, t, o, evs, role=None):
    return {"subject": s, "type": t, "role": role, "object": o, "first_reading_order": evs[0]["reading_order"],
            "last_reading_order": evs[-1]["reading_order"], "n_evidence": len(evs), "evidence": evs}  # fmt: skip


# ---- schema 的类别规则 ----


def test_category_rules():
    H, A = HUMAN, ANIMAL
    assert category_allows("KIN", H, H) and category_allows("KIN", A, A)
    assert not category_allows("KIN", A, H) and not category_allows("KIN", H, A)
    for t in ("SPOUSE", "BETROTHED", "MENTOR"):
        assert category_allows(t, H, H) and not category_allows(t, A, H) and not category_allows(t, H, A)
    assert category_allows("MEMBER_OF", H, None) and not category_allows("MEMBER_OF", A, None)
    assert category_allows("SERVES", None, None) and not category_allows("SERVES", H, A)
    assert category_allows("RULES", H, None) and not category_allows("RULES", A, None)
    assert category_allows("HOLDS_TITLE", H, None) and not category_allows("HOLDS_TITLE", A, None)
    for t in ("KILLED", "ENEMY", "ALLY", "RESIDES", "PARTICIPATES"):
        assert category_allows(t, A, H) and category_allows(t, H, A) and category_allows(t, A, None)
    assert category_allows("COMPANION", A, H) and not category_allows("COMPANION", H, A)


def test_extraction_schema_unchanged():
    """图谱 schema 升级不改变逐块抽取看到的 schema 说明（抽取版本不变，不必重新抽取）。"""
    sch = Schema(extensions=("title",))
    assert sch.extract_version == "kg-schema-1+title" and sch.version == "kg-schema-2+title"
    assert "伙伴" not in sch.prompt_relation_types()
    assert "COMPANION" not in {r.name for r in sch.chunk_relation_types}


# ---- 关系清洗 ----


def test_clean_relations_rewrites_drops_and_keeps():
    cats = person_category_of(
        [ent(ALAN), ent(MIRA), ent(STONE, category=ANIMAL), ent(PEBBLE, category=ANIMAL),
         ent(FAMILY, "Organization", category="家族"), ent(KEEP, "Location")]
    )  # fmt: skip
    assert cats[ALAN] == HUMAN and cats[STONE] == ANIMAL and cats[FAMILY] is None
    rels = [
        rel(ALAN, "KIN", STONE, [ev(2)], role="father"),  # "阿岚是石头的父亲" → 伙伴
        rel(MIRA, "SERVES", STONE, [ev(3)]),  # 方向弄反的效忠 → 伙伴，方向统一为动物 → 人
        rel(STONE, "SERVES", ALAN, [ev(1), ev(2, start=9)]),  # 与第一条同一对：合并证据
        rel(STONE, "MEMBER_OF", FAMILY, [ev(4)]),  # 动物 → 组织的成员：不允许，丢弃
        rel(STONE, "SPOUSE", MIRA, [ev(5)]),  # 不允许，丢弃
        rel(STONE, "KIN", PEBBLE, [ev(5)], role="sibling"),  # 动物之间的同窝亲属：保留
        rel(STONE, "KILLED", ALAN, [ev(6)]),  # 不限类别：保留
        rel(ALAN, "KIN", MIRA, [ev(1)], role="father"),  # 人之间：保留
        rel(ALAN, "RESIDES", KEEP, [ev(1)]),
        rel(COIN, "KIN", ALAN, [ev(7)], role="other"),  # 非角色：去掉
    ]
    stats = Counter()
    out = clean_relations(rels, cats, {COIN}, stats)
    by = {(r["subject"], r["type"], r["object"]): r for r in out}
    comp = by[(STONE, "COMPANION", ALAN)]
    assert [(e["reading_order"], e["start"]) for e in comp["evidence"]] == [(1, 0), (2, 0), (2, 9)]
    assert comp["n_evidence"] == 3 and comp["first_reading_order"] == 1 and comp["last_reading_order"] == 2
    assert comp["rewritten_from"] == {"KIN:father": 1, "SERVES": 2}
    assert (STONE, "COMPANION", MIRA) in by
    assert (STONE, "KIN", PEBBLE) in by and (STONE, "KILLED", ALAN) in by and (ALAN, "KIN", MIRA) in by
    assert (ALAN, "RESIDES", KEEP) in by
    assert not any(r["type"] in ("MEMBER_OF", "SPOUSE") and STONE in (r["subject"], r["object"]) for r in out)
    assert not any(COIN in (r["subject"], r["object"]) for r in out)
    assert stats["rel_dropped_category_MEMBER_OF"] == 1 and stats["rel_dropped_category_SPOUSE"] == 1
    assert stats["rel_dropped_non_character_KIN"] == 1 and stats["rel_companion"] == 2
    # 清洗后：人与动物之间不再有亲属 / 配偶 / 成员 / 效忠
    for r in out:
        if r["type"] in ("KIN", "SPOUSE", "MEMBER_OF", "SERVES"):
            assert cats.get(r["subject"]) == cats.get(r["object"]) or None in (cats[r["subject"]], cats[r["object"]])


def test_relations_spoiler_filter_unaffected_by_rewrite():
    """改写后的伙伴关系保留原证据的 reading_order，按阅读进度过滤照常生效。"""
    cats = {ALAN: HUMAN, STONE: ANIMAL}
    out = clean_relations([rel(STONE, "KIN", ALAN, [ev(5), ev(8)], role="other")], cats, set(), Counter())
    [r] = out
    assert r["type"] == "COMPANION" and [e["reading_order"] for e in r["evidence"]] == [5, 8]
    assert r["first_reading_order"] == 5  # 阅读进度 4 时整条关系不可见


# ---- 类别判定 ----


def test_entity_cards_take_majority_descriptions():
    ents = [ent(STONE, aliases=("小石",)), ent(ALAN, name="阿岚·铁杉", groups=("铁杉",)), ent(FAMILY, "Organization")]
    m2e = {"c1#0": STONE, "c2#0": STONE, "c3#0": STONE, "c4#0": STONE, "c1#1": ALAN, "c1#2": FAMILY}
    descs = {
        "c1#0": "阿岚的灰狼",
        "c2#0": "阿岚的灰狼",
        "c3#0": "一头狼",
        "c4#0": "阿岚附身其中看到雪原",
        "c1#1": "城主",
    }
    cards = entity_cards(ents, m2e, descs, max_descriptions=2)
    assert [c.entity_id for c in cards] == sorted([ALAN, STONE])  # 只有人物，按 ID 排序
    stone = next(c for c in cards if c.entity_id == STONE)
    assert stone.descriptions == ("阿岚的灰狼", "一头狼") and stone.aliases == ("小石",)
    assert "群体：铁杉" in next(c for c in cards if c.entity_id == ALAN).line(1)


def test_judge_categories_with_defaults():
    cards = [EntityCard(f"demo:person:p{i}", f"p{i}", (), (), ("描述",), 1) for i in range(5)]
    assert [len(b) for b in batches(cards, 2)] == [2, 2, 1]

    def responder(msgs):
        user = msgs[-1]["content"]
        items = []
        for line in user.splitlines()[1:]:
            i = int(line.split(".", 1)[0])
            name = line.split(".", 1)[1].split("｜")[0].strip()
            if name == "p1":
                items.append({"i": i, "category": "动物"})
            elif name == "p2":
                items.append({"i": i, "category": "非角色"})
            elif name == "p3":
                items.append({"i": i, "category": "外星人"})  # 不合法 → 默认人
            # p4：漏判 → 默认人
            elif name == "p0":
                items.append({"i": i, "category": "人", "reason": "城主"})
        return json.dumps({"items": items}, ensure_ascii=False)

    chat = FakeChatModel(responder)
    stats = Counter()
    j = asyncio.run(judge_categories(cards, chat, load_prompt("kg_category"), batch_size=2, concurrency=2, stats=stats))
    assert chat.calls == 3
    assert {k.rsplit(":", 1)[1]: v.category for k, v in j.items()} == {
        "p0": HUMAN, "p1": ANIMAL, "p2": NON_CHARACTER, "p3": HUMAN, "p4": HUMAN,
    }  # fmt: skip
    assert j["demo:person:p3"].source == "default" and stats["category_invalid"] == 1
    assert stats["category_missing"] == 1


def test_overrides_assign_and_drop(tmp_path):
    p = tmp_path / "aliases.toml"
    p.write_text(
        '[[category]]\nname = "石头"\ncategory = "动物"\n\n'
        f'[[category]]\nname = "{COIN}"\ncategory = "非角色"\n\n'
        '[[category]]\nname = "没有这个人"\ncategory = "人"\n',
        encoding="utf-8",
    )
    ov = Overrides.load(p)
    ents = [ent(ALAN, name="阿岚·铁杉"), ent(STONE), ent(COIN), ent(FAMILY, "Organization", category="家族")]
    from novel_rag.kg.category import Judgment

    j = {ALAN: Judgment(HUMAN, "", "llm"), STONE: Judgment(HUMAN, "", "llm"), COIN: Judgment(HUMAN, "", "llm")}
    stats = Counter()
    apply_category_overrides(j, ents, ov.categories, stats)
    assert j[STONE].category == ANIMAL and j[STONE].source == "override" and j[COIN].category == NON_CHARACTER
    assert stats["category_override_unresolved"] == 1
    dropped = assign_categories(ents, j)
    assert dropped == {COIN}
    assert [e["category"] for e in ents] == [HUMAN, ANIMAL, None, "家族"]  # 组织的类别不动
    kept, m2e = drop_entities(ents, {"a#0": ALAN, "b#0": COIN, "c#0": STONE}, dropped, stats)
    assert COIN not in {e["entity_id"] for e in kept} and m2e == {"a#0": ALAN, "c#0": STONE}
    with pytest.raises(ValueError):
        apply_category_overrides(j, ents, {"石头": "妖怪"}, Counter())


def test_no_judgment_defaults_to_human():
    ents = [ent(ALAN), ent(STONE)]
    assert assign_categories(ents, {}) == set()
    assert {e["category"] for e in ents} == {HUMAN}


# ---- 常用词名字 ----


def test_weak_names_by_linked_ratio(tmp_path):
    p = tmp_path / "main.dic"
    p.write_text("﻿夏日\n石头\n山羊\n灰岩堡\n", encoding="utf-8")
    words = load_word_list(p)
    assert words == {"夏日", "石头", "山羊", "灰岩堡"} and load_word_list(tmp_path / "none.dic") is None
    records = [
        {"mentions": [{"name": "夏日", "count": 1}, {"name": "山羊", "count": 2}, {"name": "灰岩堡", "count": 2}]},
    ]
    texts = [
        "夏日跟着阿岚。那年的夏日很长，夏日的河水很暖。",  # 夏日出现 3 次，只有 1 次是名字
        "山羊在山上，山羊是个佣兵头子。灰岩堡在北方，灰岩堡很冷。",
        "石头砸了下来。",  # 石头：没有被抽取为提及
    ]
    ents = [
        ent("demo:person:夏日"), ent("demo:person:山羊·某"), ent(KEEP, "Location"),
        ent("demo:person:阿岚·铁杉", name="阿岚·铁杉", aliases=("石头", "阿岚")),
    ]  # fmt: skip
    ents[1]["aliases"] = [{"alias": "山羊", "visible_from": 1, "first_chunk": "", "link": "x"}]
    stats = Counter()
    weak, details = weak_names(
        ents, linked_form_counts(records), CorpusCounter(texts), words, min_linked_ratio=0.5, stats=stats
    )
    assert weak == {"demo:person:夏日": ["夏日"], "demo:person:阿岚·铁杉": ["石头"]}
    d = {x["form"]: x for x in details}
    assert d["山羊"]["ratio"] == 1.0 and not d["山羊"]["weak"]  # 原文里的"山羊"都是名字
    assert d["灰岩堡"]["weak"] is False and stats["weak_forms"] == 2
    # 出现在更长的实体名字里的不算：灰岩堡主 是另一个实体的名字
    cc = CorpusCounter(["灰岩堡主来到灰岩堡。"])
    assert cc("灰岩堡") == 2 and cc("灰岩堡", ["灰岩堡主"]) == 1


ROWS = [
    {"entity_id": "demo:person:夏日", "type": "Person", "name": "夏日", "name_visible": 1, "aliases": [],
     "alias_visible": [], "first_ro": 1, "n_mentions": 30, "weak_names": ["夏日"], "category": ANIMAL},
    {"entity_id": "demo:person:阿岚·铁杉", "type": "Person", "name": "阿岚·铁杉", "name_visible": 1,
     "aliases": ["阿岚", "石头"], "alias_visible": [1, 6], "first_ro": 1, "n_mentions": 50, "weak_names": ["石头"]},
    {"entity_id": "demo:location:灰岩堡", "type": "Location", "name": "灰岩堡", "name_visible": 1,
     "aliases": [], "alias_visible": [], "first_ro": 1, "n_mentions": 9},
]  # fmt: skip


def test_linker_weak_match_and_visibility():
    lk = EntityLinker(ROWS)
    spans = lk.link("阿岚在夏日带着石头去灰岩堡", 10)
    assert [(s.text, s.weak) for s in spans] == [("阿岚", False), ("夏日", True), ("石头", True), ("灰岩堡", False)]
    assert spans[1].candidates[0].entity_id == "demo:person:夏日" and spans[1].strong_candidates == []
    # 别名可见性照常：ro 5 时"石头"还不是阿岚的可见别名
    assert [s.text for s in lk.link("石头去灰岩堡", 5)] == ["灰岩堡"]
    # 固定流程（图谱第三路）不采用弱匹配
    route = GraphRoute(queries=None, linker=lk, chunks=None)  # type: ignore[arg-type]
    assert route._entities("阿岚在夏日带着石头去灰岩堡", 10) == ["demo:person:阿岚·铁杉", "demo:location:灰岩堡"]
    assert route._weak_spans == ["夏日", "石头"]


def test_render_marks_animals_and_common_words():
    rows = [
        {"entity_id": "demo:person:夏日", "type": "Person", "name": "夏日", "qualifier": None, "category": ANIMAL,
         "aliases": ["夏日"], "common_word": True,
         "facts": [{"type": "COMPANION", "role": None, "outgoing": True, "other": "阿岚·铁杉",
                    "other_type": "Person", "other_category": HUMAN}]},
    ]  # fmt: skip
    text = render(QueryResult("resolve", {}, rows), lambda c: c, 1000)
    assert "Person/动物" in text and "夏日 —伙伴/坐骑→ 阿岚·铁杉" in text and "也是常用词" in text
    nb = [{"type": "COMPANION", "role": None, "outgoing": False, "other": "夏日", "other_id": "demo:person:夏日",
           "other_type": "Person", "other_category": ANIMAL, "n_evidence": 1,
           "evidence": [{"chunk_id": "demo-1:0001:0000", "ro": 1}]}]  # fmt: skip
    assert "夏日（动物） —伙伴/坐骑→ 该实体" in render(QueryResult("neighbors", {}, nb), lambda c: c, 1000)
