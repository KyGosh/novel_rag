"""P3.1 第 7 项：称呼归属（绰号、自称并进真实人物）。样例全部自拟。"""

import asyncio
import json
from collections import Counter
from dataclasses import dataclass

from novel_rag.kg.category import Judgment, assign_categories, entity_cards, place_name_rule
from novel_rag.kg.epithet import (
    Attribution,
    Candidate,
    EpithetJob,
    EpithetOut,
    Passage,
    PassageVerdict,
    apply_attributions,
    attribute_epithets,
    build_jobs,
    decide,
    find_raw,
    full_name_segments,
    name_forms,
    render_job,
)
from novel_rag.kg.schema import ANIMAL, EPITHET, HUMAN, NON_CHARACTER
from novel_rag.llm.fake import FakeChatModel
from novel_rag.prompts import load_prompt

ALAN, MIRA, BRAN = "demo:person:阿岚·铁杉", "demo:person:米拉·铁杉", "demo:person:布伦·灰石"
GHOST, BOAT = "demo:person:灰岩的幽灵", "demo:person:白鸥号"


@dataclass
class M:
    mid: str
    chunk_id: str
    book_id: str
    chapter_idx: int
    reading_order: int
    name: str
    count: int = 1
    description: str = ""


def ent(eid, aliases=(), name_visible=0, alias_visible=0):
    return {
        "entity_id": eid, "type": "Person", "name": eid.rsplit(":", 1)[1], "name_visible_from": name_visible,
        "groups": [], "n_mentions": 0, "n_chunks": 0, "first_reading_order": 0, "last_reading_order": 0,
        "first_chunk": "", "aliases": [{"alias": a, "visible_from": alias_visible, "first_chunk": "", "link": "x"}
                                       for a in aliases],
    }  # fmt: skip


def chunk(ch, seq):
    return f"demo-1:{ch:04d}:{seq:04d}"


# 第 1 章：米拉自称"灰岩的幽灵"（第 2 段），第 3 章再出现一次；第 1 章开头（第 0 段）只是传闻，没有指明是谁
TEXT = {
    chunk(1, 0): "人们说城里出了个灰岩的幽灵，夜里在墙头游荡。",
    chunk(1, 1): "阿岚·铁杉在大厅里等着，布伦·灰石站在他身边。",
    chunk(1, 2): "米拉·铁杉披上斗篷，对自己说：我就是灰岩的幽灵，谁也认不出我。",
    chunk(3, 0): "米拉·铁杉又一次溜出去，灰岩的幽灵今晚不会出现在墙头。",
}
MENTIONS = {
    "a0": M("a0", chunk(1, 1), "demo-1", 1, 1, "阿岚·铁杉"),
    "b0": M("b0", chunk(1, 1), "demo-1", 1, 1, "布伦·灰石"),
    "r0": M("r0", chunk(1, 2), "demo-1", 1, 1, "米拉·铁杉"),
    "r1": M("r1", chunk(3, 0), "demo-1", 3, 3, "米拉·铁杉"),
    "g0": M("g0", chunk(1, 0), "demo-1", 1, 1, "灰岩的幽灵"),
    "g1": M("g1", chunk(1, 2), "demo-1", 1, 1, "灰岩的幽灵"),
    "g2": M("g2", chunk(3, 0), "demo-1", 3, 3, "灰岩的幽灵", count=2),
    "s0": M("s0", chunk(1, 1), "demo-1", 1, 1, "白鸥号"),
}
M2E = {"a0": ALAN, "b0": BRAN, "r0": MIRA, "r1": MIRA, "g0": GHOST, "g1": GHOST, "g2": GHOST, "s0": BOAT}
CATS = {ALAN: HUMAN, BRAN: HUMAN, MIRA: HUMAN, GHOST: EPITHET, BOAT: NON_CHARACTER}


def entities():
    return [ent(ALAN), ent(MIRA, aliases=("夜莺",), alias_visible=5), ent(BRAN), ent(GHOST), ent(BOAT)]


def jobs(**kw):
    params = {"max_passages": 6, "window_chars": 10, "max_candidates": 20} | kw
    return build_jobs(entities(), M2E, MENTIONS, TEXT, CATS, **params)


def test_build_jobs_passages_and_candidates():
    (job,) = jobs()
    assert job.entity_id == GHOST
    assert [p.chunk_id for p in job.passages] == [chunk(1, 0), chunk(1, 2), chunk(3, 0)]  # 按阅读顺序
    assert all("【灰岩的幽灵】" in p.text for p in job.passages)
    ids = [c.entity_id for c in job.candidates]
    assert ids[0] == MIRA  # 段内出现的优先
    assert set(ids) == {MIRA, ALAN, BRAN}  # 不含非角色和称呼自己
    mira = next(c for c in job.candidates if c.entity_id == MIRA)
    assert "夜莺" not in mira.display  # 段落之后才可见的别名不进候选表（防剧透）
    assert len(jobs(max_passages=2)[0].passages) == 2
    assert len(jobs(max_candidates=1)[0].candidates) == 1
    msgs = render_job(job, load_prompt("kg_epithet"))
    assert "灰岩的幽灵" in msgs[-1]["content"] and "1. 米拉·铁杉" in msgs[-1]["content"]


def _job(passages, cands=(MIRA, ALAN)):
    ps = [Passage(f"demo-1:{ch:04d}:{i:04d}", "demo-1", ch, ro, "……") for i, (ch, ro) in enumerate(passages)]
    return EpithetJob(GHOST, "灰岩的幽灵", ps, [Candidate(c, c) for c in cands])


BY_CHAPTER = {("demo-1", 1): {MIRA, ALAN}, ("demo-1", 3): {MIRA}}


def out(*targets, kind="专属", other=()):
    """other：这些段号判为"指候选之外的人"（target 为 0）。"""
    return EpithetOut(
        kind=kind,
        passages=[
            PassageVerdict(p=i, target=t, same_person=True, other_person=i in other, basis="自称")
            for i, t in enumerate(targets, 1)
        ],
    )


def test_decide_rules():
    job = _job([(1, 1), (1, 2), (3, 3)])
    a = decide(job, out(0, 1, 1), BY_CHAPTER, set(), min_support=0.5)
    assert (a.status, a.target_id, a.visible_from) == ("merged", MIRA, 2)  # 可见起点是第一次被归属的段
    assert not a.generic and not a.all_chapters
    # 旧行为（alias 模式）：各段指向不同的人时放弃
    assert decide(job, out(2, 1, 1), BY_CHAPTER, set(), min_support=0.5, generic_mode="alias").status == "conflict"
    assert decide(job, out(0, 0, 1), BY_CHAPTER, set(), min_support=0.5).status == "low_support"
    assert decide(job, out(0, 0, 0), BY_CHAPTER, set(), min_support=0.5).status == "no_target"
    # 目标不在该段所在的章：作废（第 3 章没有阿岚）
    assert decide(_job([(3, 3)]), out(2), BY_CHAPTER, set(), min_support=0.5).status == "no_target"
    # 越界编号作废
    assert decide(_job([(1, 1)]), out(9), BY_CHAPTER, set(), min_support=0.5).status == "no_target"
    assert decide(job, None, BY_CHAPTER, set(), min_support=0.5).status == "llm_failed"
    # LLM 自检 same_person 为假的段作废
    not_same = EpithetOut(kind="专属", passages=[PassageVerdict(p=i, target=1, same_person=False) for i in (1, 2, 3)])
    assert decide(job, not_same, BY_CHAPTER, set(), min_support=0.5).status == "no_target"


def test_possessive_epithet_not_merged_into_owner():
    forms = name_forms(entities())
    assert {"米拉·铁杉", "米拉", "铁杉", "夜莺"} <= forms[MIRA]
    job = EpithetJob(GHOST, "米拉的随从", [Passage("c", "demo-1", 1, 1, "……")], [Candidate(MIRA, "米拉·铁杉")])
    assert decide(job, out(1), BY_CHAPTER, set(), min_support=0.5, forms_of=forms).status == "possessive"
    nick = EpithetJob(GHOST, "夜莺的随从", job.passages, job.candidates)  # 别名同样适用
    assert decide(nick, out(1), BY_CHAPTER, set(), min_support=0.5, forms_of=forms).status == "possessive"
    other = EpithetJob(GHOST, "灰岩的幽灵", job.passages, job.candidates)
    assert decide(other, out(1), BY_CHAPTER, set(), min_support=0.5, forms_of=forms).status == "merged"


def test_real_name_protection():
    segs = full_name_segments(entities())
    assert {"米拉", "铁杉", "阿岚"} <= segs
    real = EpithetJob(GHOST, "米拉", [Passage("c", "demo-1", 1, 1, "……")], [Candidate(MIRA, "米拉·铁杉")])
    assert decide(real, out(1), BY_CHAPTER, segs, min_support=0.5).status == "real_name"
    full = EpithetJob(GHOST, "米拉·白桦", real.passages, real.candidates)
    assert decide(full, out(1), BY_CHAPTER, segs, min_support=0.5).status == "real_name"


def test_apply_attributions_keeps_earlier_mentions_and_visibility():
    ents, m2e = entities(), dict(M2E)
    att = Attribution(
        GHOST,
        "灰岩的幽灵",
        MIRA,
        "merged",
        visible_from=1,
        first_chunk=chunk(1, 2),
        chapters=[["demo-1", 1], ["demo-1", 3]],
    )
    stats = Counter()
    # 第 0 段（ro 1）与归属段同 ro：按 ro 判断，一并改挂；用 ro 2 演示"更早的留在原实体"
    att_late = Attribution(
        GHOST, "灰岩的幽灵", MIRA, "merged", visible_from=3, first_chunk=chunk(3, 0), chapters=[["demo-1", 3]]
    )
    out_ents, drop = apply_attributions(ents, m2e, MENTIONS, [att_late], stats)
    assert drop == set()
    by = {e["entity_id"]: e for e in out_ents}
    assert m2e["g2"] == MIRA and m2e["g0"] == GHOST and m2e["g1"] == GHOST
    assert by[GHOST]["n_mentions"] == 2  # 更早的两次留在称呼实体
    assert by[MIRA]["n_mentions"] == 2 + 2 and by[MIRA]["last_reading_order"] == 3
    alias = next(a for a in by[MIRA]["aliases"] if a["alias"] == "灰岩的幽灵")
    assert (alias["visible_from"], alias["link"]) == (3, "epithet")
    # 全部提及都改挂后，称呼实体删除
    ents2, m2e2 = entities(), dict(M2E)
    out2, _ = apply_attributions(ents2, m2e2, MENTIONS, [att], Counter())
    assert GHOST not in {e["entity_id"] for e in out2}
    assert {m2e2[m] for m in ("g0", "g1", "g2")} == {MIRA}
    # 只改挂"有段落被归属到目标"的章里的提及：只归属了第 1 章时，第 3 章的提及留在称呼实体
    ents4, m2e4 = entities(), dict(M2E)
    only1 = Attribution(
        GHOST, "灰岩的幽灵", MIRA, "merged", visible_from=1, first_chunk=chunk(1, 2), chapters=[["demo-1", 1]]
    )
    out4 = {e["entity_id"]: e for e in apply_attributions(ents4, m2e4, MENTIONS, [only1], Counter())[0]}
    assert (m2e4["g0"], m2e4["g1"], m2e4["g2"]) == (MIRA, MIRA, GHOST) and out4[GHOST]["n_mentions"] == 2
    # 没有归属成功的不动
    out3, drop3 = apply_attributions(
        entities(), dict(M2E), MENTIONS, [Attribution(GHOST, "x", None, "no_target")], Counter()
    )
    assert GHOST in {e["entity_id"] for e in out3} and drop3 == set()  # 专属称呼归属不上：保留为孤立人物
    # 专属称呼（all_chapters）：不限于被归属的章，可见起点之后的提及全部改挂
    ents5, m2e5 = entities(), dict(M2E)
    allc = Attribution(
        GHOST, "灰岩的幽灵", MIRA, "merged", visible_from=1, first_chunk=chunk(1, 2), chapters=[["demo-1", 1]],
        all_chapters=True,
    )  # fmt: skip
    apply_attributions(ents5, m2e5, MENTIONS, [allc], Counter())
    assert {m2e5[m] for m in ("g0", "g1", "g2")} == {MIRA}


def test_attribute_epithets_end_to_end_with_fake_llm():
    def responder(msgs):
        user = msgs[-1]["content"]
        n = user.count("【灰岩的幽灵】")
        cand = next(line for line in user.splitlines() if "米拉·铁杉" in line)
        k = int(cand.split(".", 1)[0])
        # 第 1 段只是传闻，判 0；其余段指米拉
        items = [{"p": i, "target": 0 if i == 1 else k, "same_person": True, "basis": "自称"} for i in range(1, n + 1)]
        return json.dumps({"kind": "专属", "passages": items})

    chat = FakeChatModel(responder)
    stats = Counter()
    js = jobs()
    atts = asyncio.run(
        attribute_epithets(
            js,
            chat,
            load_prompt("kg_epithet"),
            M2E,
            MENTIONS,
            full_name_segments(entities()),
            min_support=0.5,
            concurrency=2,
            stats=stats,
        )  # fmt: skip
    )
    (a,) = atts
    assert (a.status, a.target_id, a.visible_from, a.first_chunk) == ("merged", MIRA, 1, chunk(1, 2))
    assert a.votes == {MIRA: 2} and stats["epithet_merged"] == 1 and chat.calls == 1
    assert a.chapters == [["demo-1", 1], ["demo-1", 3]]


def test_epithet_category_counts_as_human_when_not_attributed():
    ents = entities()

    @dataclass
    class J:
        category: str

    dropped = assign_categories(ents, {GHOST: J(EPITHET), BOAT: J(NON_CHARACTER)})
    assert dropped == {BOAT}
    assert next(e for e in ents if e["entity_id"] == GHOST)["category"] == HUMAN
    # 判定卡片照常生成（称呼也是人物实体）
    assert any(c.entity_id == GHOST for c in entity_cards(ents, M2E, {}))


def test_place_name_rule():
    ents = entities() + [
        {"entity_id": "demo:person:灰岩堡", "type": "Person", "name": "灰岩堡", "aliases": []},
        {"entity_id": "demo:location:灰岩堡", "type": "Location", "name": "灰岩堡", "aliases": [{"alias": "灰岩"}]},
        {"entity_id": "demo:person:灰岩", "type": "Person", "name": "灰岩", "aliases": []},
    ]
    j = {"demo:person:灰岩堡": Judgment(EPITHET, "", "llm"), "demo:person:灰岩": Judgment(ANIMAL, "", "llm"),
         MIRA: Judgment(HUMAN, "", "llm")}  # fmt: skip
    stats = Counter()
    place_name_rule(j, ents, stats)
    assert j["demo:person:灰岩堡"].category == NON_CHARACTER and j["demo:person:灰岩堡"].source == "rule"
    assert j["demo:person:灰岩"].category == ANIMAL  # 只改判"人"和"称呼"
    assert j[MIRA].category == HUMAN and stats["category_place_name"] == 1


def test_find_raw_maps_normalized_positions_back():
    text = "有人说：“灰岩”的幽灵 又来了。"
    i, j = find_raw(text, "灰岩的幽灵")
    assert text[i:j] == "灰岩”的幽灵"  # 区间落在原文上，引号在区间内也照常标出
    assert find_raw("前面有“引号”，后面才是灰岩的幽灵", "灰岩的幽灵") == (12, 17)
    assert find_raw("无关文字", "灰岩的幽灵") is None


def test_alias_forms_visible_from_their_own_first_appearance():
    ms = dict(MENTIONS) | {"g3": M("g3", chunk(3, 0), "demo-1", 3, 3, "幽灵小姐")}
    m2e = dict(M2E) | {"g3": GHOST}
    ents = entities()
    next(e for e in ents if e["entity_id"] == GHOST)["aliases"] = [
        {"alias": "幽灵小姐", "visible_from": 3, "first_chunk": "", "link": "x"},
        {"alias": "从未出现的写法", "visible_from": 1, "first_chunk": "", "link": "x"},
    ]
    att = Attribution(
        GHOST,
        "灰岩的幽灵",
        MIRA,
        "merged",
        visible_from=1,
        first_chunk=chunk(1, 2),
        chapters=[["demo-1", 1], ["demo-1", 3]],
    )
    stats = Counter()
    out_ents = {e["entity_id"]: e for e in apply_attributions(ents, m2e, ms, [att], stats)[0]}
    al = {a["alias"]: a for a in out_ents[MIRA]["aliases"]}
    assert al["灰岩的幽灵"]["visible_from"] == 1
    assert (al["幽灵小姐"]["visible_from"], al["幽灵小姐"]["first_chunk"]) == (3, chunk(3, 0))
    assert "从未出现的写法" not in al and stats["epithet_alias_form_skipped"] == 1


def test_generic_epithet_links_by_chapter_without_alias():
    job = _job([(1, 1), (1, 2), (3, 3)])
    # 模型判为泛称：即使各段都指米拉，也只按章挂靠，不作别名
    a = decide(job, out(1, 1, 1, kind="泛称"), BY_CHAPTER, set(), min_support=0.5)
    assert (a.status, a.generic, a.target_id, a.multi) == ("linked", True, None, False)
    assert [v[1] for v in a.verdicts] == [MIRA, MIRA, MIRA]  # 逐段原始判定留档
    assert [x[:3] for x in a.links] == [["demo-1", 1, MIRA], ["demo-1", 3, MIRA]]
    # 判为专属、但各段指向不同的人：降为泛称；第 1 章内指向不一致不挂，第 3 章挂米拉
    b = decide(job, out(2, 1, 1), BY_CHAPTER, set(), min_support=0.5)
    assert b.generic and b.multi and [x[:3] for x in b.links] == [["demo-1", 3, MIRA]]
    # 判为专属、各段都指米拉，但有一段明确指候选之外的人：降为泛称，那一章不挂
    c = decide(job, out(1, 1, 0, other=(3,)), BY_CHAPTER, set(), min_support=0.5)
    assert c.generic and c.multi and [x[:3] for x in c.links] == [["demo-1", 1, MIRA]]
    # 专属称呼、各段都指候选之外的人（主人自己不在候选里）：不算多指，按专属称呼归属不上处理（保留）
    e = decide(job, out(0, 0, 0, other=(1, 2, 3)), BY_CHAPTER, set(), min_support=0.5)
    assert (e.status, e.generic, e.multi) == ("no_target", False, False)
    # 泛称一章也挂不上
    d = decide(job, out(0, 0, 0, kind="泛称"), BY_CHAPTER, set(), min_support=0.5)
    assert (d.status, d.links) == ("generic_unlinked", [])
    # "某人的某某"的泛称：那一章不挂
    forms = name_forms(entities())
    pos = EpithetJob(GHOST, "米拉的随从", job.passages, job.candidates)
    assert decide(pos, out(1, 1, 1, kind="泛称"), BY_CHAPTER, set(), min_support=0.5, forms_of=forms).links == []
    # 真名保护先于泛称
    assert decide(EpithetJob(GHOST, "米拉", job.passages, job.candidates), out(1, 1, 1, kind="泛称"),
                  BY_CHAPTER, {"米拉"}, min_support=0.5).status == "real_name"  # fmt: skip


def test_apply_generic_links_moves_chapter_mentions_and_drops_rest():
    ents, m2e = entities(), dict(M2E)
    att = Attribution(
        GHOST, "灰岩的幽灵", None, "linked", generic=True, multi=True, links=[["demo-1", 3, MIRA, "自称"]]
    )
    stats = Counter()
    out_ents, drop = apply_attributions(ents, m2e, MENTIONS, [att], stats)
    by = {e["entity_id"]: e for e in out_ents}
    assert m2e["g2"] == MIRA and m2e["g0"] == GHOST and m2e["g1"] == GHOST  # 只挂第 3 章
    assert "灰岩的幽灵" not in {a["alias"] for a in by[MIRA]["aliases"]}  # 泛称不作别名
    assert drop == {GHOST}  # 剩下的提及连同称呼实体删掉
    assert stats["epithet_generic_mentions_linked"] == 1 and stats["epithet_generic_entities_dropped"] == 1
    # 没有证据指向不同的人（可能是只以身份为人所知的人）：剩余提及保留为孤立人物
    ents2, m2e2 = entities(), dict(M2E)
    unl = Attribution(GHOST, "灰岩的幽灵", None, "generic_unlinked", generic=True)
    out2, drop2 = apply_attributions(ents2, m2e2, MENTIONS, [unl], Counter())
    assert drop2 == set() and GHOST in {e["entity_id"] for e in out2}
    # 有证据指向不同的人、一章也没挂上：删掉
    multi = Attribution(GHOST, "灰岩的幽灵", None, "generic_unlinked", generic=True, multi=True)
    assert apply_attributions(entities(), dict(M2E), MENTIONS, [multi], Counter())[1] == {GHOST}
    # 全部提及都挂出去了：称呼实体直接去掉，不在删除名单里
    ents3, m2e3 = entities(), dict(M2E)
    full = Attribution(GHOST, "灰岩的幽灵", None, "linked", generic=True,
                       links=[["demo-1", 1, MIRA, ""], ["demo-1", 3, MIRA, ""]])  # fmt: skip
    out3, drop3 = apply_attributions(ents3, m2e3, MENTIONS, [full], Counter())
    assert GHOST not in {e["entity_id"] for e in out3} and drop3 == set()


def test_epithet_cluster_with_real_name_alias_must_match_target():
    # 称呼簇"灰岩的幽灵"自己带着别名"米拉"（全名人物的名）：只能归给名字含"米拉"的人
    ents = entities()
    next(e for e in ents if e["entity_id"] == GHOST)["aliases"] = [
        {"alias": "米拉", "visible_from": 1, "first_chunk": "", "link": "x"}
    ]
    forms, segs = name_forms(ents), full_name_segments(ents)
    job = _job([(1, 1), (1, 2)])
    assert decide(job, out(2, 2), BY_CHAPTER, segs, min_support=0.5, forms_of=forms).status == "no_target"  # 阿岚
    ok = decide(job, out(1, 1), BY_CHAPTER, segs, min_support=0.5, forms_of=forms)
    assert (ok.status, ok.target_id) == ("merged", MIRA)
