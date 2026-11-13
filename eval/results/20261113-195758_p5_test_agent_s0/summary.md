# p5_test_agent_s0

- 时间：2026-11-13T20:06:48；commit `3fcb8a787e`（有未提交改动）
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 test，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 122, 'limit_tokens': 15, 'finish_implicit': 1, 'limit_tool_calls': 3}；判分方式：{'rule': 81, 'abstain': 3, 'judge': 57}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 26424.9, 'p95': 46082.7}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.933 | 0.000 | 0.933 | 0.033 | 0.950 | 0.900 | 0.900 | 0.747 | 0.900 | 6.6 | 3.8 | 17030.8 |
| 人物关系 | 33 | 0.909 | 0.030 | 0.939 | 0.030 | 0.803 | 0.758 | 0.758 | 0.521 | 0.758 | 7.8 | 5.4 | 20254.0 |
| 多跳 | 44 | 0.864 | 0.068 | 0.932 | 0.023 | 0.710 | 0.657 | 0.517 | 0.595 | 0.657 | 7.4 | 4.9 | 20868.8 |
| 跨章节情节 | 34 | 0.676 | 0.147 | 0.824 | 0.000 | 0.730 | 0.672 | 0.632 | 0.549 | 0.672 | 8.1 | 6.1 | 24305.5 |
| **总体** | 141 | 0.844 | 0.064 | 0.908 | 0.021 | 0.788 | 0.736 | 0.683 | 0.598 | 0.736 | 7.5 | 5.1 | 20737.0 |

总体准确率 0.844，95% CI [0.7801, 0.9007]
