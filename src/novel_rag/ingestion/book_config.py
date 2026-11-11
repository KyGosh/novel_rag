"""可选的系列级书籍配置 `data/books/{series_id}/book.toml`。没有这个文件时一切按通用规则处理。

示例：
    [[volumes]]
    number = 1
    title = "某某卷名"          # 纠正目录中的卷名

    [pov]                       # 章节标题 → POV 人物；值为空字符串表示该标题不是 POV
    "某个别称标题" = "某人"

    [[sections]]                # 覆盖 section_type；volume 省略表示对所有卷生效
    volume = 1
    title = "某章标题"
    type = "back_matter"

    [kg]                        # 知识图谱的书籍专属配置（全部可选）
    extensions = ["title"]      # 启用的 schema 领域扩展（目前只有 title：头衔节点与"持有头衔"关系）
    honorifics = ["某称谓"]      # 实体名归一时额外去掉的称谓（通用称谓表在 kg/names.py）
    generic_terms = ["某泛称"]   # 额外的泛称（不建实体节点）
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KG_EXTENSIONS = ("title",)
SECTION_TYPES = ("narrative", "appendix", "front_matter", "back_matter", "dedication", "volume_title", "blank")


class BookConfigError(ValueError):
    pass


@dataclass(frozen=True)
class KGBookConfig:
    extensions: tuple[str, ...] = ()
    honorifics: tuple[str, ...] = ()
    generic_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class BookConfig:
    volume_titles: dict[int, str] = field(default_factory=dict)
    pov: dict[str, str | None] = field(default_factory=dict)
    section_types: dict[tuple[int | None, str], str] = field(default_factory=dict)
    kg: KGBookConfig = field(default_factory=KGBookConfig)
    source: str | None = None  # 配置文件路径（写进质检报告）

    def section_type_for(self, volume: int, title: str) -> str | None:
        return self.section_types.get((volume, title)) or self.section_types.get((None, title))


def load_book_config(path: Path) -> BookConfig:
    if not path.is_file():
        return BookConfig()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {"volumes", "pov", "sections", "kg"}
    if unknown:
        raise BookConfigError(f"{path} 中有未知的配置项：{sorted(unknown)}")

    volume_titles: dict[int, str] = {}
    for v in raw.get("volumes", []):
        if not isinstance(v.get("number"), int) or not isinstance(v.get("title"), str):
            raise BookConfigError(f"volumes 条目需要整数 number 和字符串 title：{v}")
        volume_titles[v["number"]] = v["title"]

    pov: dict[str, str | None] = {}
    for title, who in raw.get("pov", {}).items():
        if not isinstance(who, str):
            raise BookConfigError(f"pov 的值必须是字符串：{title} = {who!r}")
        pov[title] = who or None

    section_types: dict[tuple[int | None, str], str] = {}
    for s in raw.get("sections", []):
        st, title, vol = s.get("type"), s.get("title"), s.get("volume")
        if st not in SECTION_TYPES or not isinstance(title, str) or not (vol is None or isinstance(vol, int)):
            raise BookConfigError(f"sections 条目无效（type 取值 {SECTION_TYPES}）：{s}")
        section_types[(vol, title)] = st

    kg_raw = raw.get("kg", {})
    unknown_kg = set(kg_raw) - {"extensions", "honorifics", "generic_terms"}
    if unknown_kg:
        raise BookConfigError(f"[kg] 中有未知的配置项：{sorted(unknown_kg)}")
    lists = {}
    for key in ("extensions", "honorifics", "generic_terms"):
        v = kg_raw.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise BookConfigError(f"[kg].{key} 必须是字符串列表")
        lists[key] = tuple(v)
    bad = set(lists["extensions"]) - set(KG_EXTENSIONS)
    if bad:
        raise BookConfigError(f"未知的 schema 扩展：{sorted(bad)}（可选 {KG_EXTENSIONS}）")
    return BookConfig(volume_titles, pov, section_types, KGBookConfig(**lists), source=str(path))
