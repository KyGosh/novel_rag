# p5_test_rag

- 时间：2026-11-13T19:55:20；commit `3fcb8a787e`（有未提交改动）
- 系统：{"name": "rag", "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v1，划分 test，141 题；采样编号 0；并发 8
- 终止原因：{'answered': 141}；判分方式：{'rule': 76, 'judge': 54, 'abstain': 11}；无效引用 10 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 4967.1, 'p95': 6365.9}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 0.983 | 0.933 | 0.773 | 0.983 | 0.917 | 1.000 | 0.869 | 0.983 | 1.0 | 1.0 | 3463.1 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.030 | 0.788 | 0.742 | 0.672 | 0.788 | 0.758 | 0.924 | 0.645 | 0.818 | 1.0 | 1.0 | 3449.7 |
| 多跳 | 44 | 0.705 | 0.000 | 0.705 | 0.159 | 0.627 | 0.288 | 0.616 | 0.627 | 0.551 | 0.706 | 0.573 | 0.627 | 1.0 | 1.0 | 3429.5 |
| 跨章节情节 | 34 | 0.559 | 0.206 | 0.765 | 0.088 | 0.627 | 0.544 | 0.592 | 0.627 | 0.561 | 0.765 | 0.690 | 0.642 | 1.0 | 1.0 | 3445.6 |
| **总体** | 141 | 0.780 | 0.050 | 0.830 | 0.078 | 0.741 | 0.593 | 0.661 | 0.741 | 0.680 | 0.834 | 0.681 | 0.751 | 1.0 | 1.0 | 3445.3 |

总体准确率 0.780，95% CI [0.7092, 0.8511]
