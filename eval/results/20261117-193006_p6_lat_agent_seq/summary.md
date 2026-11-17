# p6_lat_agent_seq

- 时间：2026-11-17T19:38:18；commit `2526357ad9`（有未提交改动）
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，28 题；采样编号 0；并发 1
- 终止原因：{'finish': 25, 'finish_implicit': 1, 'limit_tokens': 2}；判分方式：{'rule': 15, 'judge': 13}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 15919.6, 'p95': 27744.2}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 7 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.762 | 1.000 | 7.0 | 3.4 | 17424.3 |
| 人物关系 | 7 | 0.857 | 0.000 | 0.857 | 0.000 | 1.000 | 1.000 | 0.857 | 0.714 | 1.000 | 8.1 | 4.9 | 20931.3 |
| 多跳 | 7 | 0.857 | 0.000 | 0.857 | 0.000 | 0.738 | 0.619 | 0.429 | 0.452 | 0.619 | 8.4 | 6.3 | 23876.3 |
| 跨章节情节 | 7 | 0.571 | 0.429 | 1.000 | 0.000 | 0.714 | 0.714 | 0.714 | 0.595 | 0.714 | 6.6 | 4.7 | 19515.1 |
| **总体** | 28 | 0.821 | 0.107 | 0.929 | 0.000 | 0.863 | 0.833 | 0.750 | 0.631 | 0.833 | 7.5 | 4.8 | 20436.8 |

总体准确率 0.821，95% CI [0.6786, 0.9643]
