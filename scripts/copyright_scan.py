"""版权自查（P6 发布前检查）：扫描入库文件和全部历史提交中的文件，找出与本地原文连续重合达到一定长度的片段。

做法：把 data/books/*/text.txt（规范化原文）去掉空白后，所有长度为 N 的窗口放进集合；对每个被扫描的文件同样去掉
空白，滑动窗口查集合，命中后向后延伸到最长的连续重合。**只报告位置和长度，不输出原文**。
- 当前入库文件：git ls-files
- 历史：git rev-list --all --objects 列出的所有 blob（去重），按 blob 报告，并给出它出现过的路径
跳过二进制文件和超过 --max-bytes 的文件。原文和报告都在本地（报告写到 data/reports/，不入库）。

用法：
    uv run python scripts/copyright_scan.py                 # 默认 N = 30
    uv run python scripts/copyright_scan.py --min-chars 20  # 更严格
退出码：有命中为 1，没有为 0。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from novel_rag.config import PROJECT_ROOT, get_settings

_WS = re.compile(r"\s+")


def git(*args: str, stdin: str | None = None) -> bytes:
    data = stdin.encode("utf-8") if stdin is not None else None
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, input=data, capture_output=True, check=True).stdout


def build_index(books_dir: Path, n: int) -> set[int]:
    idx: set[int] = set()
    for p in sorted(books_dir.glob("*/text.txt")):
        t = _WS.sub("", p.read_text(encoding="utf-8"))
        idx.update(hash(t[i : i + n]) for i in range(len(t) - n + 1))
    return idx


def scan_text(text: str, idx: set[int], n: int, full: str) -> list[dict[str, int]]:
    """返回命中区间（去掉空白后的偏移与长度）。full = 全部原文拼接（用来把命中延伸到最长）。"""
    t = _WS.sub("", text)
    hits, i = [], 0
    while i <= len(t) - n:
        if hash(t[i : i + n]) in idx and t[i : i + n] in full:
            j = i + n
            while j < len(t) and t[i : j + 1] in full:
                j += 1
            hits.append({"offset": i, "length": j - i})
            i = j
        else:
            i += 1
    return hits


def decode(b: bytes) -> str | None:
    if b"\x00" in b[:8000]:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-bytes", type=int, default=50_000_000)
    args = ap.parse_args()
    s = get_settings()
    n = args.min_chars
    full = "\n".join(_WS.sub("", p.read_text(encoding="utf-8")) for p in sorted(s.paths.books_dir.glob("*/text.txt")))
    idx = build_index(s.paths.books_dir, n)
    print(f"原文窗口 {len(idx)} 个（N = {n}）", flush=True)

    report: dict[str, object] = {"created": datetime.now().isoformat(timespec="seconds"), "min_chars": n}
    # 1. 当前入库文件
    tracked = [x for x in git("ls-files", "-z").decode("utf-8").split("\0") if x]
    cur_hits = {}
    for path in tracked:
        p = PROJECT_ROOT / path
        if not p.is_file() or p.stat().st_size > args.max_bytes:
            continue
        text = decode(p.read_bytes())
        if text is not None:
            h = scan_text(text, idx, n, full)
            if h:
                cur_hits[path] = h
    report["tracked"] = {"files": len(tracked), "hits": cur_hits}
    print(f"当前入库文件 {len(tracked)} 个，命中 {len(cur_hits)} 个", flush=True)
    # 2. 全部历史中的 blob
    paths: dict[str, set[str]] = {}
    for line in git("rev-list", "--all", "--objects").decode("utf-8").splitlines():
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(sha, set()).add(path)
    shas = sorted(paths)
    types = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", stdin=input_lines(shas)).decode()
    blobs = [
        ln.split()[0] for ln in types.splitlines() if ln.split()[1] == "blob" and int(ln.split()[2]) <= args.max_bytes
    ]
    hist_hits = {}
    for sha in blobs:
        text = decode(git("cat-file", "-p", sha))
        if text is None:
            continue
        h = scan_text(text, idx, n, full)
        if h:
            hist_hits[sha] = {"paths": sorted(paths[sha]), "hits": h}
    commits = len(git("rev-list", "--all").decode().split())
    report["history"] = {"commits": commits, "blobs": len(blobs), "hits": hist_hits}
    print(f"历史提交 {commits} 个、blob {len(blobs)} 个，命中 {len(hist_hits)} 个", flush=True)
    for sha, v in hist_hits.items():
        print(f"  {sha[:10]} {v['paths']}：" + "、".join(f"偏移 {x['offset']} 长 {x['length']}" for x in v["hits"][:5]))
    for path, h in cur_hits.items():
        print(f"  当前 {path}：" + "、".join(f"偏移 {x['offset']} 长 {x['length']}" for x in h[:5]))
    out = s.paths.reports_dir / f"copyright_scan_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"报告：{out}")
    return 1 if (cur_hits or hist_hits) else 0


def input_lines(shas: list[str]) -> str:
    return "\n".join(shas) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
