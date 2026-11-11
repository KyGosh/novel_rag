# p4_graphrag_dev

- 时间：2026-11-12T00:46:18；commit `df8bb4b9eb`
- 系统：{"name": "graphrag", "retriever": {"name": "bm25+vector+graph+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5, "max_tier": 1, "entity_types": null, "path_node_types": null, "path_exclude_node_types": null, "min_entity_mentions": 1, "kg_version": "56be6611"}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 4
- 终止原因：{'answered': 141}；判分方式：{'rule': 75, 'judge': 50, 'abstain': 16}；无效引用 2 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 2353.8, 'p95': 4562.6}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.778 | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 | 1.0 | 1.0 | 3440.2 |
| 人物关系 | 33 | 0.879 | 0.000 | 0.879 | 0.030 | 0.939 | 0.818 | 0.624 | 0.939 | 0.939 | 0.970 | 0.759 | 0.939 | 1.0 | 1.0 | 3448.5 |
| 多跳 | 44 | 0.523 | 0.000 | 0.523 | 0.273 | 0.606 | 0.326 | 0.591 | 0.606 | 0.515 | 0.689 | 0.618 | 0.629 | 1.0 | 1.0 | 3445.5 |
| 跨章节情节 | 34 | 0.412 | 0.324 | 0.735 | 0.088 | 0.515 | 0.485 | 0.555 | 0.515 | 0.471 | 0.706 | 0.659 | 0.529 | 1.0 | 1.0 | 3463.0 |
| **总体** | 141 | 0.681 | 0.078 | 0.759 | 0.114 | 0.746 | 0.623 | 0.636 | 0.746 | 0.707 | 0.825 | 0.731 | 0.756 | 1.0 | 1.0 | 3449.3 |

总体准确率 0.681，95% CI [0.5957, 0.7589]
