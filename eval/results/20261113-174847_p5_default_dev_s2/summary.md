# p5_default_dev_s2

- 时间：2026-11-13T17:57:43；commit `1349ce9b2f`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 2；并发 8
- 终止原因：{'finish': 123, 'limit_tokens': 15, 'limit_tool_calls': 3}；判分方式：{'rule': 86, 'judge': 50, 'abstain': 5}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 26740.8, 'p95': 48054.9}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 0.967 | 0.967 | 0.967 | 0.778 | 0.967 | 5.8 | 2.8 | 13769.2 |
| 人物关系 | 33 | 1.000 | 0.000 | 1.000 | 0.000 | 0.970 | 0.970 | 0.970 | 0.720 | 0.970 | 7.4 | 4.6 | 18326.0 |
| 多跳 | 44 | 0.864 | 0.023 | 0.886 | 0.091 | 0.860 | 0.780 | 0.576 | 0.713 | 0.769 | 7.6 | 5.3 | 22255.3 |
| 跨章节情节 | 34 | 0.529 | 0.353 | 0.882 | 0.029 | 0.774 | 0.696 | 0.598 | 0.563 | 0.696 | 8.9 | 6.9 | 27615.4 |
| **总体** | 141 | 0.844 | 0.092 | 0.936 | 0.035 | 0.888 | 0.844 | 0.756 | 0.692 | 0.840 | 7.5 | 5.0 | 20822.6 |

总体准确率 0.844，95% CI [0.7801, 0.9007]
