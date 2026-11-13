# p5_drift_end

- 时间：2026-11-13T20:18:47；commit `c26dd2cc7d`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，24 题；采样编号 0；并发 8
- 终止原因：{'finish': 20, 'limit_tokens': 4}；判分方式：{'rule': 13, 'judge': 11}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25621.8, 'p95': 40579.6}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.783 | 1.000 | 7.3 | 4.2 | 18371.8 |
| 人物关系 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.806 | 1.000 | 6.3 | 3.8 | 14939.2 |
| 多跳 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 0.889 | 0.889 | 0.639 | 0.861 | 0.889 | 8.3 | 5.5 | 24841.8 |
| 跨章节情节 | 6 | 0.667 | 0.333 | 1.000 | 0.000 | 0.750 | 0.667 | 0.667 | 0.528 | 0.667 | 9.7 | 7.8 | 32668.0 |
| **总体** | 24 | 0.917 | 0.083 | 1.000 | 0.000 | 0.910 | 0.889 | 0.826 | 0.744 | 0.889 | 7.9 | 5.3 | 22705.2 |

总体准确率 0.917，95% CI [0.7917, 1.0]
