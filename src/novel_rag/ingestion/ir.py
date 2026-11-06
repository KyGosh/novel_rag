"""统一中间表示：各格式解析器的共同输出，下游步骤只依赖这里的类型，不感知原始格式。

层级：Document → Part（分册：目录顶层条目，或格式自身的最外层划分）→ Section（章节）→ Block。
解析器只"如实反映文件结构"；分册合并成卷、section_type、章节编号等业务规则在 structure.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 段落文本中注释标记所在位置的占位符。规范化规则不会改动它；拼接 text.txt 时删除并记录偏移。
NOTE_MARK = "￼"

BlockKind = Literal["heading", "paragraph", "image"]
IssueLevel = Literal["error", "warning"]


@dataclass
class NoteRef:
    """已从正文剥离的注释：标记在所属块文本中以 NOTE_MARK 表示，按出现顺序与 notes 列表一一对应。"""

    label: str  # 原标记文本，如 "[1]"、"①"
    text: str  # 注释内容（不含编号）
    marker_locator: str
    note_locator: str


@dataclass
class Block:
    kind: BlockKind
    text: str = ""
    level: int = 0  # 标题层级 1–6；其他块为 0
    locator: str = ""  # 来源定位：epub 中为 "文件路径#元素id" 或 "文件路径@元素序号"
    image_src: str | None = None  # 插图在容器内的资源路径
    notes: list[NoteRef] = field(default_factory=list)


@dataclass
class Section:
    """章节。title 为 None 表示分册中第一个章节标题之前的内容（标题页、献词、插图页等）。"""

    title: str | None
    locator: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Part:
    """分册。title 为 None 表示目录第一项之前的内容（封面、作者介绍等）。"""

    title: str | None
    locator: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Issue:
    level: IssueLevel
    code: str
    message: str
    locator: str = ""


@dataclass
class Document:
    source_name: str
    source_sha256: str
    format: str
    parser_version: str
    metadata: dict[str, str] = field(default_factory=dict)
    parts: list[Part] = field(default_factory=list)
    resources: dict[str, bytes] = field(default_factory=dict)  # 图片等二进制资源，键为容器内路径
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)  # 解析器自身的统计（注释配对、spine 归类等），供质检报告

    def warn(self, code: str, message: str, locator: str = "") -> None:
        self.issues.append(Issue("warning", code, message, locator))

    def error(self, code: str, message: str, locator: str = "") -> None:
        self.issues.append(Issue("error", code, message, locator))


class IngestError(RuntimeError):
    """无法继续解析的输入错误（DRM、超出解压限制、不支持的格式等）。"""
