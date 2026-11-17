"""版权自查（P6 发布前检查）：扫描入库文件和全部历史提交中的文件，找出与本地原文连续重合达到一定长度的片段。

做法：把 data/books/*/text.txt（规范化原文）去掉空白后，所有长度为 N 的窗口放进集合；对每个被扫描的文件同样去掉
空白，滑动窗口查集合，命中后向后延伸到最长的连续重合（src/novel_rag/eval/redact.py）。**只报告位置和长度，不输出原文**。
- 当前入库文件：git ls-files
- 历史：git rev-list --all --objects 列出的所有 blob（去重），按 blob 报告，并给出它出现过的路径
跳过二进制文件和超过 --max-bytes 的文件。原文和报告都在本地（报告写到 data/reports/，不入库）。

--fix：对命中的入库 JSON / JSONL 文件，把每个字符串里与原文重合的片段替换成"[…]"后写回（其余内容逐字不变），
再重新扫描。历史提交不改写（2026-11-17 用户决定：只处理当前文件）。

用法：
    uv run python scripts/copyright_scan.py                 # 默认 N = 30
    uv run python scripts/copyright_scan.py --fix --no-history
退出码：（处理后）当前文件仍有命中为 1，否则为 0；历史中的命中只报告。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime

from novel_rag.config import PROJECT_ROOT, get_settings
from novel_rag.eval.redact import RedactIndex, find_overlaps, strip_ws


def git(*args: str, stdin: str | None = None) -> bytes:
    data = stdin.encode("utf-8") if stdin is not None else None
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, input=data, capture_output=True, check=True).stdout


def scan_text(text: str, idx: set[int], n: int, full: str) -> list[dict[str, int]]:
    """返回命中区间（去掉空白后的偏移与长度）。full = 全部原文拼接（用来把命中延伸到最长）。"""
    t, _ = strip_ws(text)
    return [{"offset": o, "length": ln} for o, ln in find_overlaps(t, idx, n, full)]


def decode(b: bytes) -> str | None:
    if b"\x00" in b[:8000]:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fix_file(path: str, ri: RedactIndex) -> bool:
    """JSONL 逐行、JSON 整体：只替换字符串里的重合片段，序列化方式与写结果时一致（ensure_ascii=False）。"""
    p = PROJECT_ROOT / path
    raw = p.read_text(encoding="utf-8")
    before = ri.count
    if path.endswith(".jsonl"):
        lines = []
        for line in raw.splitlines():
            if not line:
                continue
            obj = json.loads(line)
            new = ri.redact_obj(obj)
            lines.append(line if new == obj else json.dumps(new, ensure_ascii=False))
        out = "".join(x + "\n" for x in lines)
    elif path.endswith(".json"):
        out = json.dumps(ri.redact_obj(json.loads(raw)), ensure_ascii=False, indent=2) + "\n"
    else:
        return False
    if ri.count > before:
        p.write_text(out, encoding="utf-8", newline="")
        return True
    return False


def scan_tracked(ri: RedactIndex, max_bytes: int) -> tuple[list[str], dict[str, list[dict[str, int]]]]:
    tracked = [x for x in git("ls-files", "-z").decode("utf-8").split("\0") if x]
    hits = {}
    for path in tracked:
        p = PROJECT_ROOT / path
        if not p.is_file() or p.stat().st_size > max_bytes:
            continue
        text = decode(p.read_bytes())
        if text is not None:
            h = scan_text(text, ri.idx, ri.n, ri.full)
            if h:
                hits[path] = h
    return tracked, hits


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-bytes", type=int, default=50_000_000)
    ap.add_argument("--fix", action="store_true", help="替换当前入库 JSON / JSONL 文件里的重合片段")
    ap.add_argument("--no-history", action="store_true", help="不扫描历史提交")
    args = ap.parse_args()
    s = get_settings()
    ri = RedactIndex((p.read_text(encoding="utf-8") for p in sorted(s.paths.books_dir.glob("*/text.txt"))),
                     args.min_chars)  # fmt: skip
    print(f"原文窗口 {len(ri.idx)} 个（N = {ri.n}）", flush=True)
    report: dict[str, object] = {"created": datetime.now().isoformat(timespec="seconds"), "min_chars": ri.n}

    tracked, cur_hits = scan_tracked(ri, args.max_bytes)
    print(f"当前入库文件 {len(tracked)} 个，命中 {len(cur_hits)} 个", flush=True)
    if args.fix and cur_hits:
        fixed = [path for path in cur_hits if fix_file(path, ri)]
        print(f"已处理 {len(fixed)} 个文件、替换 {ri.count} 处", flush=True)
        report["fixed"] = {"files": fixed, "replaced": ri.count}
        tracked, cur_hits = scan_tracked(ri, args.max_bytes)
        print(f"处理后重新扫描：命中 {len(cur_hits)} 个", flush=True)
    report["tracked"] = {"files": len(tracked), "hits": cur_hits}
    for path, h in cur_hits.items():
        print(f"  当前 {path}：" + "、".join(f"偏移 {x['offset']} 长 {x['length']}" for x in h[:5]))

    if not args.no_history:
        paths: dict[str, set[str]] = {}
        for line in git("rev-list", "--all", "--objects").decode("utf-8").splitlines():
            sha, _, path = line.partition(" ")
            if path:
                paths.setdefault(sha, set()).add(path)
        types = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                    stdin="\n".join(sorted(paths)) + "\n").decode()  # fmt: skip
        blobs = [x.split()[0] for x in types.splitlines()
                 if x.split()[1] == "blob" and int(x.split()[2]) <= args.max_bytes]  # fmt: skip
        hist_hits = {}
        for sha in blobs:
            text = decode(git("cat-file", "-p", sha))
            if text is not None and (h := scan_text(text, ri.idx, ri.n, ri.full)):
                hist_hits[sha] = {"paths": sorted(paths[sha]), "hits": h}
        commits = len(git("rev-list", "--all").decode().split())
        report["history"] = {"commits": commits, "blobs": len(blobs), "hits": hist_hits}
        print(f"历史提交 {commits} 个、blob {len(blobs)} 个，命中 {len(hist_hits)} 个", flush=True)
        for sha, v in hist_hits.items():
            spans = "、".join(f"偏移 {x['offset']} 长 {x['length']}" for x in v["hits"][:5])
            print(f"  {sha[:10]} {v['paths']}：{spans}")
    out = s.paths.reports_dir / f"copyright_scan_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"报告：{out}")
    return 1 if cur_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
