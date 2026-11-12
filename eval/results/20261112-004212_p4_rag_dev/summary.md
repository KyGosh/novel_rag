# p4_rag_dev

- 时间：2026-11-12T00:44:10；commit `df8bb4b9eb`
- 系统：{"name": "rag", "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 4
- 终止原因：{'answered': 141}；判分方式：{'rule': 75, 'judge': 51, 'abstain': 15}；无效引用 2 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 2778.3, 'p95': 3828.0}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.794 | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 | 1.0 | 1.0 | 3439.7 |
| 人物关系 | 33 | 0.909 | 0.000 | 0.909 | 0.030 | 0.955 | 0.803 | 0.645 | 0.955 | 0.955 | 0.985 | 0.774 | 0.955 | 1.0 | 1.0 | 3453.1 |
| 多跳 | 44 | 0.500 | 0.000 | 0.500 | 0.295 | 0.606 | 0.311 | 0.597 | 0.606 | 0.515 | 0.678 | 0.617 | 0.617 | 1.0 | 1.0 | 3444.9 |
| 跨章节情节 | 34 | 0.471 | 0.353 | 0.824 | 0.029 | 0.544 | 0.529 | 0.569 | 0.544 | 0.485 | 0.706 | 0.667 | 0.559 | 1.0 | 1.0 | 3464.5 |
| **总体** | 141 | 0.695 | 0.085 | 0.780 | 0.106 | 0.756 | 0.625 | 0.649 | 0.756 | 0.714 | 0.825 | 0.736 | 0.764 | 1.0 | 1.0 | 3450.4 |

总体准确率 0.695，95% CI [0.617, 0.773]
