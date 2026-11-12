# p4_lat2_rag

- 时间：2026-11-12T18:08:55；commit `839f2be161`
- 系统：{"name": "rag", "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v1，划分 dev，28 题；采样编号 0；并发 1
- 终止原因：{'answered': 28}；判分方式：{'rule': 12, 'judge': 12, 'abstain': 4}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 2693.7, 'p95': 3526.4}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 7 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.952 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.0 | 1.0 | 3440.6 |
| 人物关系 | 7 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.857 | 0.691 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 | 1.0 | 1.0 | 3398.4 |
| 多跳 | 7 | 0.286 | 0.000 | 0.286 | 0.571 | 0.548 | 0.143 | 0.417 | 0.548 | 0.548 | 0.667 | 0.595 | 0.548 | 1.0 | 1.0 | 3427.6 |
| 跨章节情节 | 7 | 0.571 | 0.143 | 0.714 | 0.000 | 0.643 | 0.571 | 0.559 | 0.643 | 0.643 | 0.786 | 0.905 | 0.643 | 1.0 | 1.0 | 3459.6 |
| **总体** | 28 | 0.714 | 0.036 | 0.750 | 0.143 | 0.798 | 0.643 | 0.707 | 0.798 | 0.798 | 0.863 | 0.798 | 0.798 | 1.0 | 1.0 | 3431.5 |

总体准确率 0.714，95% CI [0.5357, 0.8571]
