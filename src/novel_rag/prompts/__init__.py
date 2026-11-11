"""版本化 prompt 模板（版本号写进缓存 key 和实验记录）。

文件命名为 `{name}.v{N}.md`，放在本目录下；`load_prompt(name)` 默认取最高版本。改动 prompt 时新建下一个版本的文件，
旧版本保留，以便复现历史实验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
_NAME = re.compile(r"^(?P<name>[a-z0-9_]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def version_id(self) -> str:
        return f"{self.name}.v{self.version}"

    def sections(self) -> dict[str, str]:
        """按 `## system` / `## user` 等二级标题分节，返回 {节名: 内容}。"""
        parts: dict[str, str] = {}
        current = None
        for line in self.text.splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                parts[current] = ""
            elif current is not None:
                parts[current] += line + "\n"
        return {k: v.strip() for k, v in parts.items()}

    def render(self, **values: str) -> list[dict[str, str]]:
        """system + user 两条消息；{name} 占位符逐个替换（不用 str.format，原文里的花括号不受影响）。
        先替换 system（只放 schema 说明这类固定内容），再替换 user，原文中恰好出现的占位符不会被二次替换。"""
        sec = self.sections()
        system, user = sec["system"], sec["user"]
        for k, v in values.items():
            system = system.replace("{" + k + "}", v)
        for k, v in values.items():
            user = user.replace("{" + k + "}", v)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_prompt(name: str, version: int | None = None, directory: Path = PROMPTS_DIR) -> Prompt:
    found: dict[int, Path] = {}
    for p in directory.iterdir():
        m = _NAME.match(p.name)
        if m and m.group("name") == name:
            found[int(m.group("version"))] = p
    if not found:
        raise FileNotFoundError(f"{directory} 下没有 prompt {name!r}")
    v = max(found) if version is None else version
    if v not in found:
        raise FileNotFoundError(f"prompt {name!r} 没有 v{v}，现有版本 {sorted(found)}")
    return Prompt(name, v, found[v].read_text(encoding="utf-8"))
