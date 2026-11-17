# p6_replaytime_par_r1

- 时间：2026-11-17T18:42:12；commit `3c4961ef49`（有未提交改动）
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，28 题；采样编号 0；并发 1
- 终止原因：{'finish': 25, 'limit_tool_calls': 1, 'error': 1, 'limit_tokens': 1}；判分方式：{'rule': 16, 'none': 11, 'abstain': 1}；无效引用 0 个；出错 1 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 1250.4, 'p95': 2145.6}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 7 | 0.857 | 0.000 | 0.857 | 0.000 | 1.000 | 1.000 | 1.000 | 0.691 | 1.000 | 5.9 | 2.9 | 14418.4 |
| 人物关系 | 7 | 0.571 | 0.000 | 0.571 | 0.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 7.6 | 5.1 | 20451.6 |
| 多跳 | 7 | 0.714 | 0.000 | 0.714 | 0.143 | 0.667 | 0.667 | 0.417 | 0.444 | 0.667 | 7.0 | 4.7 | 20897.6 |
| 跨章节情节 | 7 | 0.143 | 0.000 | 0.143 | 0.000 | 0.786 | 0.786 | 0.786 | 0.643 | 0.786 | 7.4 | 5.3 | 22222.0 |
| **总体** | 28 | 0.571 | 0.000 | 0.571 | 0.036 | 0.870 | 0.870 | 0.815 | 0.574 | 0.870 | 7.0 | 4.5 | 19497.4 |

总体准确率 0.571，95% CI [0.3929, 0.75]
