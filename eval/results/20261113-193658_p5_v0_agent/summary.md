# p5_v0_agent

- 时间：2026-11-13T19:45:18；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v0，划分 all，120 题；采样编号 0；并发 8
- 终止原因：{'finish': 99, 'limit_tool_calls': 5, 'limit_tokens': 14, 'finish_implicit': 1, 'limit_rounds': 1}；判分方式：{'judge': 92, 'rule': 26, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 28710.8, 'p95': 50270.7}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.900 | 0.100 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.850 | 1.000 | 6.5 | 3.4 | 15677.6 |
| 人物关系 | 30 | 0.800 | 0.100 | 0.900 | 0.000 | 0.772 | 0.706 | 0.706 | 0.512 | 0.706 | 7.5 | 5.3 | 19979.4 |
| 多跳 | 30 | 0.833 | 0.033 | 0.867 | 0.067 | 0.883 | 0.800 | 0.617 | 0.756 | 0.800 | 7.4 | 4.8 | 20422.5 |
| 跨章节情节 | 30 | 0.433 | 0.367 | 0.800 | 0.000 | 0.556 | 0.511 | 0.328 | 0.263 | 0.450 | 9.5 | 8.2 | 32049.6 |
| **总体** | 120 | 0.742 | 0.150 | 0.892 | 0.017 | 0.803 | 0.754 | 0.662 | 0.593 | 0.739 | 7.7 | 5.4 | 22032.3 |

总体准确率 0.742，95% CI [0.6583, 0.8167]
