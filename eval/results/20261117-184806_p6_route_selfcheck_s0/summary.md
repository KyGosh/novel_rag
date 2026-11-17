# p6_route_selfcheck_s0

- 时间：2026-11-17T18:49:50；commit `1e151af114`（有未提交改动）
- 系统：{"name": "cascade_selfcheck_dry", "signal": "selfcheck", "dry": true, "stage1": {"name": "rag", "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash", "answer_prompt": "qa_answer_selfcheck.v1"}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'answered': 141}；判分方式：{'rule': 71, 'judge': 48, 'abstain': 22}；无效引用 9 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 5477.9, 'p95': 6296.8}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.767 | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 | 1.0 | 1.0 | 3590.2 |
| 人物关系 | 33 | 0.849 | 0.000 | 0.849 | 0.061 | 0.955 | 0.742 | 0.667 | 0.955 | 0.955 | 0.985 | 0.774 | 0.955 | 1.0 | 1.0 | 3594.8 |
| 多跳 | 44 | 0.477 | 0.000 | 0.477 | 0.364 | 0.606 | 0.299 | 0.571 | 0.606 | 0.515 | 0.678 | 0.617 | 0.617 | 1.0 | 1.0 | 3591.0 |
| 跨章节情节 | 34 | 0.441 | 0.324 | 0.765 | 0.118 | 0.544 | 0.426 | 0.536 | 0.544 | 0.485 | 0.706 | 0.667 | 0.559 | 1.0 | 1.0 | 3612.7 |
| **总体** | 141 | 0.667 | 0.078 | 0.745 | 0.156 | 0.756 | 0.583 | 0.638 | 0.756 | 0.714 | 0.825 | 0.736 | 0.764 | 1.0 | 1.0 | 3597.0 |

总体准确率 0.667，95% CI [0.5816, 0.7447]
