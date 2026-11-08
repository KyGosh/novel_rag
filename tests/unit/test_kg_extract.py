"""知识图谱：schema 映射、名称归一、逐块抽取的校验（证据定位、泛称、补端点、类型约束、全名采信）。样例全部自拟。"""

from novel_rag.corpus import BookText
from novel_rag.kg.extract import (
    AliasOut,
    ChunkExtractionOut,
    EntityOut,
    RelationOut,
    appears,
    full_name_supported,
    validate_extraction,
)
from novel_rag.kg.names import NameRules, normalize_name, segments
from novel_rag.kg.schema import Schema, entity_type_of, is_symmetric, map_relation
from tests.unit.book_factory import make_book

RULES = NameRules.with_extra(honorifics=["卡汗"])
SCHEMA = Schema(extensions=("title",))
CORE = Schema()

PARA = (
    "阿岚·铁杉爵士站在灰岩堡的城墙上。他的女儿米拉跑过来，喊他父亲。"
    "人称老灰狼的阿岚点点头，说铁杉家族世代镇守灰岩堡。"
    "那个骑士没有说话。阿岚是北境守护。"
)


def _book_and_chunk(tmp_path):
    make_book(tmp_path, "demo-1", [("序章", ["前面的一段话，阿岚刚回到城堡。"]), ("米拉", [PARA])], first_idx=0)
    book = BookText.load(tmp_path / "demo-1")
    ch = book.chapter(1)
    s = book.text.index(PARA)
    chunk = {
        "chunk_id": "demo-1:0001:0000",
        "book_id": "demo-1",
        "chapter_idx": 1,
        "reading_order": 1,
        "section_type": "narrative",
        "char_start": s,
        "char_end": s + len(PARA),
        "text": PARA,
        "chapter_title": ch.title,
        "pov": ch.pov,
    }
    return book, chunk


def test_map_relation_kin_roles_and_flip():
    assert map_relation("亲属", "父亲") == map_relation("父亲", None)
    m = map_relation("亲属", "女儿")
    assert m.type == "KIN" and m.role == "parent" and m.flip  # "A 是 B 的女儿" → "B 是 A 的父母"
    assert map_relation("亲属", "侄子").role == "uncle_aunt" and map_relation("亲属", "侄子").flip
    assert map_relation("配偶", None).type == "SPOUSE" and is_symmetric("SPOUSE", None)
    assert is_symmetric("KIN", "sibling") and not is_symmetric("KIN", "father")
    assert map_relation("杀死", None).type == "KILLED"
    assert map_relation("暗恋", None) is None  # 白名单外：丢弃
    assert map_relation("亲属", "某种不认识的称谓") is None
    assert map_relation("敌对关系", None).type == "ENEMY"


def test_entity_type_labels():
    assert entity_type_of("人物") == ("Person", None)
    assert entity_type_of("家族") == ("Organization", "家族")  # 类别误写成类型
    assert entity_type_of("动物") == (None, None)


def test_schema_extensions():
    assert "Title" in SCHEMA.entity_types and "Title" not in CORE.entity_types
    assert SCHEMA.allows("HOLDS_TITLE") and not CORE.allows("HOLDS_TITLE")
    assert "持有头衔" in SCHEMA.prompt_relation_types() and "持有头衔" not in CORE.prompt_relation_types()
    assert SCHEMA.version != CORE.version


def test_name_normalization_and_honorifics():
    assert normalize_name("阿岚•铁杉") == "阿岚·铁杉" == normalize_name(" 阿岚・铁杉 ")
    assert segments("阿岚·铁杉") == ("阿岚", "铁杉")
    assert RULES.canonical("阿岚·铁杉爵士") == "阿岚·铁杉"
    assert RULES.canonical("学士欧文") == "欧文"
    assert RULES.canonical("国王之手") == "国王之手"  # 去掉前缀后以"之"开头：整体是称号
    assert RULES.canonical("灰狼之王") == "灰狼之王"
    assert RULES.canonical("托戈卡汗") == "托戈"  # 书籍配置追加的称谓
    assert RULES.canonical("爵士") == "爵士"  # 去掉后不足两个字：保留


def test_generic_terms():
    for g in ("那个骑士", "老人", "他们", "卫兵们", "国王", "父亲大人"):
        assert RULES.is_generic(g), g
    for n in ("阿岚·铁杉", "米拉", "灰岩堡"):
        assert not RULES.is_generic(n), n


def test_appears_and_full_name_support():
    assert appears("阿岚·铁杉", "阿岚•铁杉站着")
    assert appears("阿岚·铁杉", "阿岚铁杉站着")  # 原文省略间隔号
    assert appears("铁杉家族", "铁杉家的人")  # 组织名补了后缀
    assert not appears("米拉", "阿岚站着")
    assert full_name_supported("阿岚", "阿岚·铁杉", "阿岚站着", "", RULES)  # 有共同片段
    assert full_name_supported("老灰狼", "阿岚·铁杉", "老灰狼来了", "上文提到阿岚", RULES)  # 片段出现在上文
    assert not full_name_supported("老灰狼", "阿岚·铁杉", "老灰狼来了", "", RULES)  # 凭空补全：不采信


def test_validate_extraction(tmp_path):
    book, chunk = _book_and_chunk(tmp_path)
    out = ChunkExtractionOut(
        entities=[
            EntityOut(name="阿岚·铁杉", type="人物", group="铁杉家族", description="灰岩堡的主人"),
            EntityOut(name="米拉", full_name="米拉·铁杉", type="人物", description="阿岚的女儿"),
            EntityOut(name="灰岩堡", type="地点"),
            EntityOut(name="那个骑士", type="人物"),  # 泛称
            EntityOut(name="林恩", type="人物"),  # 不在本段
            EntityOut(name="北境守护", type="头衔"),
            EntityOut(name="冰原", type="动物"),  # 类型不合法
            EntityOut(name="老灰狼", full_name="欧林·白桦", type="人物"),  # 全名与原文无关：不采信
        ],
        relations=[
            RelationOut(
                subject="米拉", relation="亲属", role="女儿", object="阿岚·铁杉", evidence="他的女儿米拉跑过来"
            ),
            RelationOut(subject="铁杉家族", relation="统治", object="灰岩堡", evidence="铁杉家族世代镇守灰岩堡"),
            RelationOut(subject="阿岚·铁杉", relation="持有头衔", object="北境守护", evidence="阿岚是北境守护"),
            RelationOut(subject="阿岚·铁杉", relation="杀害", object="米拉", evidence="原文里没有这句话"),
            RelationOut(subject="阿岚·铁杉", relation="暗恋", object="米拉", evidence="他的女儿米拉跑过来"),
            RelationOut(subject="灰岩堡", relation="居住", object="阿岚·铁杉", evidence="站在灰岩堡的城墙上"),
            RelationOut(subject="林恩", relation="盟友", object="阿岚·铁杉", evidence="阿岚点点头"),
        ],
        aliases=[
            AliasOut(entity="阿岚·铁杉", alias="老灰狼", evidence="人称老灰狼的阿岚点点头"),
            AliasOut(entity="阿岚·铁杉", alias="雪狐", evidence="人称雪狐"),  # 别称不在本段
        ],
    )
    rec, drops = validate_extraction(out, book, chunk, "", SCHEMA, RULES, fuzzy_threshold=90)
    names = {m["name"]: m for m in rec["mentions"]}
    assert set(names) == {"阿岚·铁杉", "米拉", "灰岩堡", "北境守护", "老灰狼", "铁杉家族"}
    assert names["铁杉家族"]["implicit"] and names["铁杉家族"]["type"] == "Organization"  # 关系端点补上
    assert names["米拉"]["full_name"] == "米拉·铁杉"
    assert names["老灰狼"]["full_name"] is None and names["老灰狼"]["claimed_full_name"] == "欧林·白桦"
    rels = {
        (rec["mentions"][r["s"]]["name"], r["type"], r["role"], rec["mentions"][r["o"]]["name"])
        for r in rec["relations"]
    }
    assert ("阿岚·铁杉", "KIN", "parent", "米拉") in rels  # "女儿"翻转
    assert ("铁杉家族", "RULES", None, "灰岩堡") in rels
    assert ("阿岚·铁杉", "HOLDS_TITLE", None, "北境守护") in rels
    assert len(rels) == 3
    for r in rec["relations"]:
        s, e = r["evidence"]
        assert chunk["char_start"] <= s < e <= chunk["char_end"]
    assert [a["alias"] for a in rec["aliases"]] == ["老灰狼"]
    assert drops["entity_generic"] == 1 and drops["entity_name_not_in_text"] == 1 and drops["entity_bad_type"] == 1
    assert drops["relation_evidence_not_found"] == 1 and drops["relation_unmapped"] == 1
    assert drops["relation_type_mismatch"] == 1 and drops["relation_unknown_entity"] == 1
    assert drops["alias_not_in_text"] == 1 and drops["full_name_unverified"] == 1

    # 不启用头衔扩展时：头衔实体与"持有头衔"关系都丢弃
    rec2, drops2 = validate_extraction(out, book, chunk, "", CORE, RULES, fuzzy_threshold=90)
    assert all(m["type"] != "Title" for m in rec2["mentions"])
    assert drops2["entity_extension_disabled"] == 1 and drops2["relation_extension_disabled"] == 1
