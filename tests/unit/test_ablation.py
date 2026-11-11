"""消融表：按名称找最新的实验记录、配对差值、Markdown 渲染。"""

import json

import pytest

from novel_rag.eval.ablation import build_table, find_run, render_markdown

CFG = {"bootstrap_samples": 500, "bootstrap_seed": 1, "ci_level": 0.95}


def _write(results, stamp, name, values, commit="abc"):
    d = results / f"{stamp}_{name}"
    d.mkdir(parents=True)
    rows = [
        {"id": f"q{i}", "type": "fact", "metrics": {"recall@5": v, "recall@20": v, "mrr": v}}
        for i, v in enumerate(values)
    ]
    mean = sum(values) / len(values)
    metrics = {
        "overall": {"n": len(values), "recall@5": mean, "recall@20": mean, "mrr": mean},
        "by_type": {"fact": {"n": len(values), "recall@5": mean}},
        "ci": {"overall": {"recall@5": [0.0, 1.0]}},
        "latency_ms": {"p50": 10.0, "p95": 20.0},
    }
    run = {"git": {"commit": commit, "dirty": False}, "dataset": {"split": "dev"}}
    (d / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (d / "per_question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_find_run_picks_latest_exact_name(tmp_path):
    _write(tmp_path, "20260101-000000", "base", [0.0, 1.0])
    newest = _write(tmp_path, "20260102-000000", "base", [1.0, 1.0])
    _write(tmp_path, "20260103-000000", "base_x", [0.0, 0.0])
    assert find_run(tmp_path, "base") == newest
    with pytest.raises(FileNotFoundError):
        find_run(tmp_path, "nope")


def test_build_table_paired_diff_and_render(tmp_path):
    _write(tmp_path, "20260101-000000", "base", [0.0] * 10)
    _write(tmp_path, "20260101-000001", "better", [1.0] * 5 + [0.0] * 5)
    spec = {"title": "t", "rows": [
        {"group": "g", "label": "基线", "run": "base"},
        {"group": "g", "label": "改进", "run": "better", "ref": "base"},
    ]}  # fmt: skip
    table = build_table(spec, tmp_path, CFG)
    d = table["rows"][1]["diff"]["recall@5"]
    assert d["diff"] == 0.5 and d["lo"] > 0
    assert "cov@4000" not in table["rows"][1]["diff"]  # 两边都没有该指标时跳过
    md = render_markdown(table)
    assert "+0.500" in md and "↑" in md and "改进" in md and "20260101-000001_better" in md
    # 按题型的配对差值：样例里全是 fact
    assert table["rows"][1]["diff_by_type"]["fact"]["recall@5"]["diff"] == 0.5
    assert "按题型的配对差值" in md

    _write(tmp_path, "20260101-000002", "other_q", [1.0] * 3)
    with pytest.raises(ValueError):
        build_table({"rows": [{"group": "g", "label": "x", "run": "other_q", "ref": "base"}]}, tmp_path, CFG)
