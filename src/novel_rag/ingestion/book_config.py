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
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SECTION_TYPES = ("narrative", "appendix", "front_matter", "back_matter", "dedication", "volume_title", "blank")


class BookConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BookConfig:
    volume_titles: dict[int, str] = field(default_factory=dict)
    pov: dict[str, str | None] = field(default_factory=dict)
    section_types: dict[tuple[int | None, str], str] = field(default_factory=dict)
    source: str | None = None  # 配置文件路径（写进质检报告）

    def section_type_for(self, volume: int, title: str) -> str | None:
        return self.section_types.get((volume, title)) or self.section_types.get((None, title))


def load_book_config(path: Path) -> BookConfig:
    if not path.is_file():
        return BookConfig()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {"volumes", "pov", "sections"}
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

    return BookConfig(volume_titles, pov, section_types, source=str(path))
