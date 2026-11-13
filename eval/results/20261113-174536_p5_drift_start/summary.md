# p5_drift_start

- 时间：2026-11-13T17:47:48；commit `bbff5482c7`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，24 题；采样编号 0；并发 4
- 终止原因：{'finish': 18, 'limit_tokens': 6}；判分方式：{'rule': 11, 'judge': 13}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 15808.8, 'p95': 30053.5}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 6 | 0.833 | 0.000 | 0.833 | 0.000 | 0.833 | 0.833 | 0.833 | 0.722 | 0.833 | 6.3 | 3.0 | 13998.3 |
| 人物关系 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 7.7 | 4.5 | 18331.8 |
| 多跳 | 6 | 0.833 | 0.000 | 0.833 | 0.000 | 1.000 | 0.889 | 0.639 | 0.792 | 0.889 | 9.0 | 6.5 | 27807.8 |
| 跨章节情节 | 6 | 0.333 | 0.667 | 1.000 | 0.000 | 0.750 | 0.667 | 0.500 | 0.389 | 0.667 | 10.0 | 8.5 | 33608.0 |
| **总体** | 24 | 0.750 | 0.167 | 0.917 | 0.000 | 0.896 | 0.847 | 0.743 | 0.684 | 0.847 | 8.2 | 5.6 | 23436.5 |

总体准确率 0.750，95% CI [0.5833, 0.9167]
