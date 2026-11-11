"""把图谱导出文件一键写进 Neo4j（先删除同一系列的旧数据，再批量写入，最后核对数量）。

用法：
    uv run python scripts/load_kg.py                      # 最近构建的图谱版本
    uv run python scripts/load_kg.py --version 1a2b3c4d
需要 Neo4j 容器已启动（docker compose up -d）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from novel_rag.config import get_settings
from novel_rag.kg.build import load_export
from novel_rag.kg.common import kg_root
from novel_rag.kg.neo4j_store import Neo4jKGStore


def latest_kg_dir(root):
    cands = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("没有图谱导出，先运行 scripts/build_kg.py")
    return cands[-1].parent


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="图谱版本，默认取最近构建的")
    args = parser.parse_args()

    s = get_settings()
    root = kg_root(s.paths.data_dir)
    d = root / args.version if args.version else latest_kg_dir(root)
    tables, manifest = load_export(d)
    store = Neo4jKGStore.connect(s.neo4j)
    t0 = time.perf_counter()
    try:
        counts = store.replace(manifest["series_id"], tables, manifest, s.kg.neo4j_batch)
    finally:
        store.close()
    expected = {
        "entities": len(tables["entities"]),
        "events": len(tables["events"]),
        "chunks": len(tables["chunks"]),
        "relations": len(tables["relations"]),
        "mentions": len(tables["mentions"]),
    }
    ok = all(counts[k] == v for k, v in expected.items())
    print(
        json.dumps(
            {
                "kg_version": manifest["kg_version"],
                "neo4j": counts,
                "export": expected,
                "ok": ok,
                "seconds": round(time.perf_counter() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        )  # fmt: skip
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
