"""实体名的规则归一：统一间隔号、去空白、去称谓；泛称判定。

通用称谓与泛称表写在这里（中文小说普遍适用）；书籍专属的称谓、泛称放在书籍配置 [kg] 中，由调用方合并。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_INTERPUNCT = re.compile(r"[•‧・･∙⋅·．・]")
_SPACE = re.compile(r"\s+")
_QUOTES = re.compile(r"[\"'“”‘’「」『』《》〈〉]")

# 称谓：后缀（"某某爵士"）与前缀（"国王某某"）。只在去掉后仍剩 ≥ 2 个字时才去掉
HONORIFIC_SUFFIXES = (
    "大学士", "学士", "爵士", "大人", "夫人", "小姐", "少爷", "老爷", "师傅", "修士", "修女", "王子", "公主", "国王",
    "王后", "女王", "陛下", "殿下", "公爵", "伯爵", "侯爵", "子爵", "男爵", "将军", "队长", "长官", "主教", "先生",
    "女士", "太太", "阁下", "亲王", "总督",
)  # fmt: skip
HONORIFIC_PREFIXES = (
    "国王", "王后", "女王", "王子", "公主", "学士", "大学士", "修士", "修女", "爵士", "公爵", "伯爵", "侯爵", "大人",
    "夫人", "老爷", "小姐", "女士", "先生", "亲王", "陛下", "殿下",
)  # fmt: skip

GENERIC_TERMS = frozenset(
    (
        "他",
        "她",
        "它",
        "他们",
        "她们",
        "它们",
        "我",
        "你",
        "您",
        "我们",
        "你们",
        "咱们",
        "自己",
        "某人",
        "那人",
        "这人",
        "有人",
        "众人",
        "大家",
        "别人",
        "对方",
        "谁",
        "老人",
        "老头",
        "老妇",
        "老太婆",
        "男孩",
        "女孩",
        "孩子",
        "小孩",
        "男人",
        "女人",
        "姑娘",
        "小伙子",
        "少年",
        "少女",
        "婴儿",
        "小子",
        "家伙",
        "丫头",
        "骑士",
        "士兵",
        "卫兵",
        "守卫",
        "哨兵",
        "侍从",
        "仆人",
        "侍女",
        "女仆",
        "奶妈",
        "学士",
        "修士",
        "修女",
        "国王",
        "王后",
        "女王",
        "王子",
        "公主",
        "大人",
        "夫人",
        "小姐",
        "父亲",
        "母亲",
        "爸爸",
        "妈妈",
        "哥哥",
        "姐姐",
        "弟弟",
        "妹妹",
        "兄弟",
        "姐妹",
        "儿子",
        "女儿",
        "丈夫",
        "妻子",
        "叔叔",
        "舅舅",
        "姑姑",
        "祖父",
        "祖母",
        "爷爷",
        "奶奶",
        "陌生人",
        "敌人",
        "朋友",
        "主人",
        "老爷",
        "少爷",
        "领主",
        "封臣",
        "王",
        "公爵",
        "伯爵",
        "侯爵",
        "爵士",
        "神",
        "诸神",
        "陛下",
        "殿下",
        "大学士",
        "囚犯",
        "俘虏",
        "信使",
        "商人",
        "农夫",
        "渔夫",
        "铁匠",
        "厨子",
        "马夫",
        "车夫",
        "船长",
        "水手",
        "歌手",
        "妓女",
        "乞丐",
        "盗贼",
        "强盗",
        "刺客",
        "巫师",
        "女巫",
        "祭司",
        "僧侣",
        "猎人",
        "屠夫",
        "母亲大人",
        "父亲大人",
        "夫君",
        "新娘",
        "新郎",
        "寡妇",
        "孤儿",
        "私生子",
        "侏儒",
        "巨人",
        "城堡",
        "城市",
        "村庄",
        "森林",
        "河流",
        "大海",
        "山",
        "北方",
        "南方",
        "东方",
        "西方",
        "王国",
        "王座",
        "军队",
        "舰队",
        "议会",
        "宫廷",
        "家族",
    )  # fmt: skip
)
_GENERIC_PREFIX = re.compile(r"^(那个|这个|一个|那位|这位|某个|某位|一位|另一个|另一位|那些|这些|几个|两个|三个)")


def normalize_name(name: str) -> str:
    """统一间隔号、去掉空白和引号。不做 NFKC（会改动全角标点）。"""
    n = _QUOTES.sub("", name or "")
    n = _SPACE.sub("", n)
    n = _INTERPUNCT.sub("·", n)
    return n.strip("·")


def segments(name: str) -> tuple[str, ...]:
    return tuple(s for s in normalize_name(name).split("·") if s)


@dataclass(frozen=True)
class NameRules:
    """称谓与泛称表（通用 + 书籍配置追加）。"""

    suffixes: tuple[str, ...] = HONORIFIC_SUFFIXES
    prefixes: tuple[str, ...] = HONORIFIC_PREFIXES
    generic: frozenset[str] = GENERIC_TERMS

    @classmethod
    def with_extra(cls, honorifics: Iterable[str] = (), generic_terms: Iterable[str] = ()) -> NameRules:
        extra = tuple(honorifics)
        by_len = lambda xs: tuple(sorted(set(xs), key=len, reverse=True))  # noqa: E731 —— 长的先匹配
        return cls(
            by_len(HONORIFIC_SUFFIXES + extra), by_len(HONORIFIC_PREFIXES + extra), GENERIC_TERMS | set(generic_terms)
        )

    def strip_honorifics(self, name: str) -> str:
        """去掉称谓（可能有多个，如"某某爵士大人"）；去掉后不足 2 个字则保留原样。"""
        n = normalize_name(name)
        changed = True
        while changed:
            changed = False
            for suf in self.suffixes:
                # "某某之王""某某的女王"这类称号整体是别名，不拆
                if n.endswith(suf) and len(n) - len(suf) >= 2 and n[-len(suf) - 1] not in "之的":
                    n, changed = n[: -len(suf)].rstrip("·"), True
                    break
            for pre in self.prefixes:
                if n.startswith(pre) and len(n) - len(pre) >= 2 and n[len(pre)] not in "之的":
                    n, changed = n[len(pre) :].lstrip("·"), True
                    break
        return n

    def canonical(self, name: str) -> str:
        return self.strip_honorifics(name)

    def fingerprint(self) -> str:
        """规则内容的 hash（写进图谱版本：改称谓 / 泛称表会改变消歧结果）。"""
        import hashlib

        blob = "|".join(["/".join(self.suffixes), "/".join(self.prefixes), "/".join(sorted(self.generic))])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def is_generic(self, name: str) -> bool:
        """代词、泛称（"老人""卫兵"）、"那个骑士"式的指称、复数，以及去掉称谓后只剩泛称的名字。"""
        n = normalize_name(name)
        if len(n) < 2:
            return True
        if n in self.generic or _GENERIC_PREFIX.match(n) or n.endswith("们"):
            return True
        return self.strip_honorifics(n) in self.generic
