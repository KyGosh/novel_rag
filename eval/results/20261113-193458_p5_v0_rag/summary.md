# p5_v0_rag

- 时间：2026-11-13T19:36:29；commit `4b884b98fe`
- 系统：{"name": "rag", "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v0，划分 all，120 题；采样编号 0；并发 8
- 终止原因：{'answered': 120}；判分方式：{'judge': 90, 'rule': 23, 'abstain': 7}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 4184.2, 'p95': 5978.2}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.933 | 0.067 | 1.000 | 0.000 | 1.000 | 1.000 | 0.878 | 1.000 | 1.000 | 1.000 | 0.859 | 1.000 | 1.0 | 1.0 | 3391.3 |
| 人物关系 | 30 | 0.833 | 0.100 | 0.933 | 0.000 | 0.806 | 0.656 | 0.600 | 0.806 | 0.706 | 0.889 | 0.723 | 0.839 | 1.0 | 1.0 | 3395.9 |
| 多跳 | 30 | 0.700 | 0.000 | 0.700 | 0.167 | 0.717 | 0.372 | 0.573 | 0.717 | 0.667 | 0.783 | 0.727 | 0.717 | 1.0 | 1.0 | 3411.1 |
| 跨章节情节 | 30 | 0.267 | 0.367 | 0.633 | 0.067 | 0.356 | 0.289 | 0.279 | 0.356 | 0.233 | 0.614 | 0.342 | 0.383 | 1.0 | 1.0 | 3456.0 |
| **总体** | 120 | 0.683 | 0.133 | 0.817 | 0.058 | 0.719 | 0.579 | 0.588 | 0.719 | 0.651 | 0.822 | 0.663 | 0.735 | 1.0 | 1.0 | 3413.6 |

总体准确率 0.683，95% CI [0.6, 0.7583]
