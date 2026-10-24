"""P0 环境检查：逐项报告 ES / Milvus / Neo4j / GPU / Embedding / Reranker / Kimi 是否可用。

用法：
    uv run python scripts/check_env.py                 # 全部检查
    uv run python scripts/check_env.py --only es,neo4j # 只查部分
结果同时写入 data/reports/check_env.json（含 IK 分词基线）。测试句子均为自拟，不含原文。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from novel_rag.config import get_settings

IK_SENTENCE = "琼恩·雪诺站在长城之上，望着北方幽暗的森林，守夜人的誓言在心中回响。"
TRANSLITERATED_NAMES = ["琼恩·雪诺", "琼恩", "雪诺"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class CheckFailed(AssertionError):
    pass


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


# ---------------- 各项检查：返回 (摘要, 详情)，失败时抛异常 ----------------


def check_es() -> tuple[str, dict[str, Any]]:
    from elasticsearch import Elasticsearch

    cfg = get_settings().es
    es = Elasticsearch(cfg.url, request_timeout=cfg.request_timeout)
    version = es.info()["version"]["number"]
    plugins = [p["component"] for p in es.cat.plugins(format="json")]
    expect("analysis-ik" in plugins, f"未安装 IK 插件，已装插件：{plugins}")

    def analyze(analyzer: str) -> list[str]:
        return [t["token"] for t in es.indices.analyze(analyzer=analyzer, text=IK_SENTENCE)["tokens"]]

    max_word, smart = analyze("ik_max_word"), analyze("ik_smart")
    multi_char = [t for t in max_word if len(t) >= 2]
    expect(len(multi_char) >= 3, f"ik_max_word 没有切出中文词：{max_word}")
    names_kept = {n: n in max_word for n in TRANSLITERATED_NAMES}
    baseline = "被切碎（符合预期，P3 回灌词典）" if not any(names_kept.values()) else f"部分成词：{names_kept}"
    return (
        f"ES {version} + IK；ik_max_word 切出 {len(max_word)} 个 token；音译人名{baseline}",
        {"version": version, "ik_max_word": max_word, "ik_smart": smart, "names_as_single_token": names_kept},
    )


def check_milvus() -> tuple[str, dict[str, Any]]:
    import numpy as np
    from pymilvus import DataType, MilvusClient

    cfg = get_settings().milvus
    client = MilvusClient(uri=cfg.uri, token=cfg.token.get_secret_value(), db_name=cfg.db_name)
    name = f"p0_smoke_{int(time.time())}"
    dim = 8
    try:
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("reading_order", DataType.INT64)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
        index = client.prepare_index_params()
        index.add_index("vec", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 64})
        client.create_collection(name, schema=schema, index_params=index)

        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((3, dim)).astype("float32")
        rows = [{"id": i, "reading_order": i + 1, "vec": vecs[i].tolist()} for i in range(3)]
        client.insert(name, rows)
        client.flush(name)
        client.load_collection(name)

        hits = client.search(name, data=[vecs[2].tolist()], limit=3, output_fields=["reading_order"])[0]
        expect(hits[0]["id"] == 2, f"最近邻应为自身，实际：{hits}")
        # 防剧透过滤的最小验证：reading_order <= 2 时看不到 id=2
        filtered = client.search(name, data=[vecs[2].tolist()], limit=3, filter="reading_order <= 2")[0]
        expect(all(h["id"] != 2 for h in filtered) and len(filtered) == 2, f"标量过滤失效：{filtered}")
        version = client.get_server_version()
    finally:
        if client.has_collection(name):
            client.drop_collection(name)
    expect(not client.has_collection(name), "临时 collection 未删除")
    return f"Milvus {version}：建表→插入→搜索→标量过滤→删除 全部成功", {"version": version}


def check_neo4j() -> tuple[str, dict[str, Any]]:
    from neo4j import GraphDatabase

    cfg = get_settings().neo4j
    with GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password.get_secret_value())) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS v, edition",
            database_=cfg.database,
        )
        server = f"{records[0]['name']} {records[0]['v']} {records[0]['edition']}"
        records, _, _ = driver.execute_query("RETURN apoc.version() AS v", database_=cfg.database)
        apoc = records[0]["v"]
    return f"{server}，APOC {apoc}", {"server": server, "apoc": apoc}


def check_gpu() -> tuple[str, dict[str, Any]]:
    from novel_rag.llm.bge import gpu_info

    info = gpu_info()
    expect(bool(info["cuda_available"]), f"torch 看不到 GPU（可能装成了 CPU 版）：{info}")
    return f"{info['name']}，显存 {info['total_mem_gb']} GB，torch {info['torch']}", info


def check_embedding() -> tuple[str, dict[str, Any]]:
    import numpy as np

    from novel_rag.llm import get_embedder

    t0 = time.perf_counter()
    emb = get_embedder()
    load_s = time.perf_counter() - t0
    q = emb.embed_query("北境的冬天来了。")
    docs = emb.embed_documents(["临冬城的城墙很高。", "今天的午饭是面条。"])
    norm = float(np.linalg.norm(q))
    device = getattr(emb, "device", "unknown")
    expect(q.shape == (1024,), f"维度应为 1024，实际 {q.shape}")
    expect(abs(norm - 1) < 1e-3, f"范数应约为 1，实际 {norm}")
    expect(docs.shape == (2, 1024), f"文档编码形状错误：{docs.shape}")
    expect(device.startswith("cuda"), f"模型不在 GPU 上：{device}")
    return (
        f"{emb.model_name} dim={q.shape[0]} norm={norm:.4f} device={device}，加载 {load_s:.1f}s",
        {"model": emb.model_name, "dim": int(q.shape[0]), "norm": norm, "device": device, "load_s": load_s},
    )


def check_reranker() -> tuple[str, dict[str, Any]]:
    from novel_rag.llm import get_reranker

    rr = get_reranker()
    query = "谁是那座北方城堡的领主？"
    relevant = "那座北方城堡由史塔克家族世代统治，现任领主是艾德公爵。"
    irrelevant = "今天的天气很好，适合去公园散步。"
    s_rel, s_irr = rr.score(query, [relevant, irrelevant])
    device = getattr(rr, "device", "unknown")
    expect(s_rel > s_irr, f"相关候选得分应更高：{s_rel:.4f} vs {s_irr:.4f}")
    expect(device.startswith("cuda"), f"模型不在 GPU 上：{device}")
    return (
        f"{rr.model_name} 相关 {s_rel:.4f} > 无关 {s_irr:.4f}，device={device}",
        {"relevant": s_rel, "irrelevant": s_irr, "device": device},
    )


class _Capital(BaseModel):
    country: str
    capital: str


def check_kimi() -> tuple[str, dict[str, Any]]:
    from novel_rag.llm.cache import CachedChatModel, open_disk_cache
    from novel_rag.llm.openai_chat import ConcurrencyLimiter, OpenAICompatChatModel
    from novel_rag.llm.usage import UsageTracker

    settings = get_settings()
    expect(bool(settings.llm.api_key.get_secret_value()), "未设置 LLM__API_KEY")
    task_cfg = settings.llm.tasks["smoke"]
    tracker = UsageTracker()
    raw = OpenAICompatChatModel("smoke", task_cfg, settings.llm, tracker, ConcurrencyLimiter(2))
    with tempfile.TemporaryDirectory() as tmp:  # 用临时缓存，保证每次检查都真正请求一次
        model = CachedChatModel(raw, open_disk_cache(Path(tmp)), tracker)
        plain = model.chat([{"role": "user", "content": "用一句话介绍你自己。"}])
        expect(bool(plain.text.strip()), "普通对话返回为空")

        msgs = [{"role": "user", "content": "法国的首都是哪里？用 JSON 回答。"}]
        s1 = model.chat(msgs, schema=_Capital, prompt_version="p0-smoke-v1")
        expect(isinstance(s1.parsed, _Capital), "结构化输出未通过校验")
        s2 = model.chat(msgs, schema=_Capital, prompt_version="p0-smoke-v1")
        expect(s2.cached and not s1.cached, "相同请求第二次没有命中缓存")
        expect(s2.parsed == s1.parsed, "缓存还原的结构化对象不一致")
        model.store.close()  # type: ignore[attr-defined]

    recs = tracker.records
    real = [r for r in recs if not r.cached]
    expect(len(recs) == 3 and len(real) == 2, f"用量记录条数不对：{recs}")
    expect(all(r.prompt_tokens > 0 and r.completion_tokens > 0 for r in real), f"token 数缺失：{real}")
    return (
        f"{task_cfg.model}（{settings.llm.structured_mode}）对话/结构化/缓存/用量均正常；"
        f"结构化 attempts={s1.attempts}，parsed={s1.parsed.model_dump()}",
        {"plain_reply": plain.text[:200], "usage": tracker.summary(), "structured_attempts": s1.attempts},
    )


CHECKS: dict[str, Callable[[], tuple[str, dict[str, Any]]]] = {
    "es": check_es,
    "milvus": check_milvus,
    "neo4j": check_neo4j,
    "gpu": check_gpu,
    "embedding": check_embedding,
    "reranker": check_reranker,
    "kimi": check_kimi,
}


def run(names: list[str], verbose: bool = False) -> list[CheckResult]:
    results = []
    for name in names:
        t0 = time.perf_counter()
        try:
            summary, details = CHECKS[name]()
            res = CheckResult(name, True, summary, details)
        except Exception as e:  # noqa: BLE001 —— 环境检查要把任何失败都报告出来
            res = CheckResult(name, False, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
            if verbose:
                print(res.details["traceback"], file=sys.stderr)
        res.details["elapsed_s"] = round(time.perf_counter() - t0, 2)
        print(f"[{'PASS' if res.ok else 'FAIL'}] {name:<10} {res.summary}", flush=True)
        results.append(res)
    return results


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 管道默认 GBK
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help=f"逗号分隔，可选：{','.join(CHECKS)}")
    parser.add_argument("-v", "--verbose", action="store_true", help="失败时打印完整 traceback")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        parser.error(f"未知检查项：{unknown}")

    results = run(names, args.verbose)
    out = get_settings().paths.data_dir / "reports" / "check_env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    failed = [r.name for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过" + (f"，失败：{failed}" if failed else ""))
    print(f"报告：{out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
