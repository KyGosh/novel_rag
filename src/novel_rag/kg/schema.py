"""知识图谱 schema：通用核心（任何小说都适用）+ 可选领域扩展（由书籍配置启用）。

- 实体类型：Person / Organization / Location / Event（Event 来自章节摘要，不逐 chunk 抽取）；扩展 title 增加 Title。
  家族不单独设类型，作为 category="家族" 的 Organization。
- 关系白名单：模型输出的说法先经同义词表映射到白名单类型（亲属再映射到角色），映射不了的丢弃并计数，
  不允许关系类型无限增长。
- 对称关系（配偶、婚约、敌对、盟友、兄弟姐妹等）按实体 ID 排序存成一个方向；"儿子/女儿"这类子辈称谓翻转成
  "客体是主体的父母"，所以亲属关系只存长辈 → 晚辈或对称两种方向。
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "kg-schema-1"

PERSON, ORG, LOC, EVENT, TITLE = "Person", "Organization", "Location", "Event", "Title"
ENTITY_LABELS = {PERSON: "人物", ORG: "组织", LOC: "地点", EVENT: "事件", TITLE: "头衔"}
LABEL_TO_TYPE = {v: k for k, v in ENTITY_LABELS.items()}
ORG_CATEGORIES = ("家族", "军团", "宗教", "行会", "政权", "其他")
EXTENSIONS = {"title": (TITLE,)}  # 扩展名 → 新增的实体类型


@dataclass(frozen=True)
class RelationType:
    name: str  # 内部类型名（Neo4j 中统一边类型 REL 的 type 属性）
    label: str  # 中文名，prompt 中展示
    subject_types: tuple[str, ...]
    object_types: tuple[str, ...]
    symmetric: bool = False
    extension: str | None = None  # 属于哪个领域扩展；None 为通用核心
    source: str = "chunk"  # chunk = 逐块抽取；summary = 来自章节摘要
    hint: str = ""  # prompt 中的说明


RELATION_TYPES: tuple[RelationType, ...] = (
    RelationType("KIN", "亲属", (PERSON,), (PERSON,), hint="主体是客体的<role>，role 必填"),
    RelationType("SPOUSE", "配偶", (PERSON,), (PERSON,), symmetric=True, hint="主体与客体是夫妻"),
    RelationType("BETROTHED", "婚约", (PERSON,), (PERSON,), symmetric=True, hint="主体与客体订有婚约"),
    RelationType(
        "SERVES", "效忠", (PERSON, ORG), (PERSON, ORG),
        hint="主体效忠、侍奉或受雇于客体（封臣、骑士、侍从、护卫、仆人、学士等），客体可以是人物或组织",
    ),
    RelationType("MENTOR", "师徒", (PERSON,), (PERSON,), hint="主体是客体的老师或师傅"),
    RelationType("MEMBER_OF", "成员", (PERSON, ORG), (ORG,), hint="主体（人物或组织）属于客体（组织），包括出身的家族"),
    RelationType("ENEMY", "敌对", (PERSON, ORG), (PERSON, ORG), symmetric=True, hint="主体与客体敌对、交战或有仇"),
    RelationType("ALLY", "盟友", (PERSON, ORG), (PERSON, ORG), symmetric=True, hint="主体与客体结盟或并肩作战"),
    RelationType("KILLED", "杀害", (PERSON, ORG), (PERSON,), hint="原文明确写出主体杀死了客体"),
    RelationType("LOCATED_IN", "位于", (LOC, ORG), (LOC,), hint="主体（地点或组织）位于客体（地点）"),
    RelationType(
        "RULES", "统治", (PERSON, ORG), (LOC, ORG),
        hint="主体（人物或组织）统治、领有或镇守客体（地点），或统领客体（组织）",
    ),
    RelationType("RESIDES", "居住", (PERSON,), (LOC,), hint="主体（人物）居住或驻留在客体（地点）"),
    RelationType(
        "HOLDS_TITLE", "持有头衔", (PERSON,), (TITLE,), extension="title",
        hint="主体（人物）持有客体（头衔），如某人担任某职位",
    ),
    RelationType("PARTICIPATES", "参与", (PERSON,), (EVENT,), source="summary"),
    RelationType("OCCURS_AT", "发生于", (EVENT,), (LOC,), source="summary"),
    RelationType("CAUSES", "导致", (EVENT,), (EVENT,), source="summary"),
)  # fmt: skip
REL_BY_NAME = {r.name: r for r in RELATION_TYPES}

# 亲属角色：(规范角色, 是否翻转)。翻转表示"主体是客体的儿子" → "客体是主体的父母"
KIN_ROLES = {
    "father": "父亲",
    "mother": "母亲",
    "parent": "父母",
    "sibling": "兄弟姐妹",
    "grandparent": "祖父母",
    "uncle_aunt": "叔伯姑舅姨",
    "cousin": "堂表亲",
    "other": "其他亲属",
}
SYMMETRIC_KIN_ROLES = {"sibling", "cousin", "other"}
KIN_PROMPT_ROLES = (
    "父亲", "母亲", "儿子", "女儿", "兄弟", "姐妹", "祖父母", "孙辈", "叔伯姑舅姨", "侄甥", "堂表亲", "其他亲属",
)  # fmt: skip
_KIN_ROLE_SYNONYMS: dict[str, tuple[str, bool]] = {}
for _words, _value in (
    (("父亲", "生父", "父", "爸爸", "养父"), ("father", False)),
    (("母亲", "生母", "母", "妈妈", "养母"), ("mother", False)),
    (("父母", "双亲"), ("parent", False)),
    (("儿子", "女儿", "子女", "孩子", "长子", "次子", "幼子", "长女", "次女", "幼女", "私生子", "私生女", "养子",
      "养女", "儿女", "子嗣", "继承人"), ("parent", True)),
    (("兄弟", "哥哥", "弟弟", "兄长", "姐妹", "姐姐", "妹妹", "兄弟姐妹", "胞兄", "胞弟", "胞姐", "胞妹",
      "同父异母", "同母异父", "孪生", "双胞胎", "长兄", "长姐"), ("sibling", False)),
    (("祖父", "祖母", "外祖父", "外祖母", "祖父母", "曾祖父", "曾祖母", "爷爷", "奶奶", "外公", "外婆"),
     ("grandparent", False)),
    (("孙子", "孙女", "孙辈", "外孙", "外孙女", "曾孙", "曾孙女"), ("grandparent", True)),
    (("叔叔", "伯父", "叔父", "叔伯", "舅舅", "舅父", "姑姑", "姑妈", "姨妈", "阿姨", "叔伯姑舅姨", "姑姨",
      "婶婶", "舅妈", "伯伯"), ("uncle_aunt", False)),
    (("侄子", "侄女", "外甥", "外甥女", "侄甥", "侄儿"), ("uncle_aunt", True)),
    (("堂兄", "堂弟", "堂姐", "堂妹", "表兄", "表弟", "表姐", "表妹", "堂兄弟", "表兄弟", "堂表亲", "表亲",
      "堂亲", "堂姐妹", "表姐妹"), ("cousin", False)),
    (("其他亲属", "亲戚", "亲属", "家人", "远亲", "姻亲"), ("other", False)),
):  # fmt: skip
    for _w in _words:
        _KIN_ROLE_SYNONYMS[_w] = _value

# 关系说法 → 白名单类型。亲属称谓本身（"父亲"等）也可以直接作为 relation 出现
_RELATION_SYNONYMS: dict[str, str] = {}
for _words, _name in (
    (("亲属", "家人", "亲戚", "血亲"), "KIN"),
    (("配偶", "夫妻", "婚姻", "妻子", "丈夫", "夫妇", "结婚"), "SPOUSE"),
    (("婚约", "订婚", "未婚妻", "未婚夫", "许配"), "BETROTHED"),
    (("效忠", "主从", "侍奉", "服侍", "封臣", "侍从", "护卫", "仆人", "臣属", "受雇", "效力", "服务", "宣誓效忠"),
     "SERVES"),
    (("师徒", "老师", "师傅", "导师", "教导"), "MENTOR"),
    (("成员", "隶属", "属于", "出身", "加入"), "MEMBER_OF"),
    (("敌对", "敌人", "仇敌", "交战", "仇人", "对手"), "ENEMY"),
    (("盟友", "结盟", "同盟", "联盟", "并肩作战"), "ALLY"),
    (("杀害", "杀死", "杀", "杀掉", "处死", "处决", "谋杀", "刺杀"), "KILLED"),
    (("位于", "坐落"), "LOCATED_IN"),
    (("统治", "领有", "镇守", "管辖", "治理", "占领"), "RULES"),
    (("居住", "住在", "驻留", "居于"), "RESIDES"),
    (("持有头衔", "头衔", "担任", "任职"), "HOLDS_TITLE"),
):  # fmt: skip
    for _w in _words:
        _RELATION_SYNONYMS[_w] = _name


@dataclass(frozen=True)
class MappedRelation:
    type: str
    role: str | None  # 只有 KIN 有角色
    flip: bool  # True 表示主客体需要对调


def map_relation(relation: str, role: str | None) -> MappedRelation | None:
    """模型输出的 (relation, role) → 白名单类型与规范角色；映射不了返回 None。"""
    rel = (relation or "").strip()
    rl = (role or "").strip()
    if rel in _KIN_ROLE_SYNONYMS and rel not in _RELATION_SYNONYMS:  # relation 直接写成了亲属称谓
        rl, rel = rel, "亲属"
    name = _RELATION_SYNONYMS.get(rel)
    if name is None and rel.endswith("关系"):
        name = _RELATION_SYNONYMS.get(rel[:-2])
    if name is None:
        return None
    if name != "KIN":
        return MappedRelation(name, None, False)
    if rl not in _KIN_ROLE_SYNONYMS:
        return MappedRelation("KIN", "other", False) if not rl else None
    canon, flip = _KIN_ROLE_SYNONYMS[rl]
    return MappedRelation("KIN", canon, flip)


# 主客体类型不符时的改写：(关系, 客体类型) → 应改成的关系。例如"持有头衔 → 某组织"其实是成员关系
TYPE_COERCIONS = {("HOLDS_TITLE", ORG): "MEMBER_OF", ("MEMBER_OF", TITLE): "HOLDS_TITLE"}


def entity_type_of(label: str) -> tuple[str | None, str | None]:
    """模型输出的类型标签 → (实体类型, 组织类别)。把组织类别（"家族""政权"）误写成类型时归为组织。"""
    t = (label or "").strip()
    if t in LABEL_TO_TYPE:
        return LABEL_TO_TYPE[t], None
    if t in ENTITY_LABELS:
        return t, None
    if t in ORG_CATEGORIES:
        return ORG, t
    return None, None


def is_symmetric(rel_type: str, role: str | None) -> bool:
    if rel_type == "KIN":
        return role in SYMMETRIC_KIN_ROLES
    return REL_BY_NAME[rel_type].symmetric


@dataclass(frozen=True)
class Schema:
    """某本书启用的 schema：通用核心 + 书籍配置中启用的扩展。"""

    extensions: tuple[str, ...] = ()

    @property
    def entity_types(self) -> tuple[str, ...]:
        extra = tuple(t for e in self.extensions for t in EXTENSIONS.get(e, ()))
        return (PERSON, ORG, LOC, *extra)  # Event 来自摘要，不在逐块抽取中出现

    @property
    def chunk_relation_types(self) -> tuple[RelationType, ...]:
        return tuple(
            r for r in RELATION_TYPES if r.source == "chunk" and (r.extension is None or r.extension in self.extensions)
        )

    def allows(self, rel_type: str) -> bool:
        r = REL_BY_NAME.get(rel_type)
        return r is not None and (r.extension is None or r.extension in self.extensions)

    @property
    def version(self) -> str:
        return SCHEMA_VERSION + ("+" + "+".join(sorted(self.extensions)) if self.extensions else "")

    def prompt_entity_types(self) -> str:
        lines = [
            "人物：有名字的人（也包括有名字的动物伙伴）",
            "组织：家族、军团、宗教、行会、政权等，category 取 " + "/".join(ORG_CATEGORIES),
            "地点：城堡、城市、地区、河流、山脉等有专名的地方",
        ]
        if "title" in self.extensions:
            lines.append("头衔：职位或爵位称号本身（如某某之手、某某总司令），只有原文写明有人持有它时才抽")
        return "\n".join(f"  - {x}" for x in lines)

    def prompt_relation_types(self) -> str:
        lines = []
        for r in self.chunk_relation_types:
            hint = r.hint
            if r.name == "KIN":
                hint += "，取 " + "、".join(KIN_PROMPT_ROLES) + " 之一"
            lines.append(f"  - {r.label}：{hint}")
        return "\n".join(lines)
