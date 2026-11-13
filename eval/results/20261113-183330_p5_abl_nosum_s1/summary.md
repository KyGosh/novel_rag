# p5_abl_nosum_s1

- 时间：2026-11-13T18:41:52；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 1；并发 8
- 终止原因：{'finish': 124, 'limit_tokens': 16, 'limit_tool_calls': 1}；判分方式：{'rule': 85, 'judge': 55, 'abstain': 1}；无效引用 4 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25299.6, 'p95': 44006.2}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 1.000 | 0.967 | 0.967 | 0.772 | 0.967 | 6.0 | 3.2 | 13244.3 |
| 人物关系 | 33 | 0.970 | 0.030 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.712 | 1.000 | 7.2 | 4.7 | 17379.2 |
| 多跳 | 44 | 0.841 | 0.023 | 0.864 | 0.023 | 0.826 | 0.761 | 0.557 | 0.605 | 0.761 | 7.4 | 5.1 | 19816.6 |
| 跨章节情节 | 34 | 0.735 | 0.206 | 0.941 | 0.000 | 0.814 | 0.760 | 0.691 | 0.559 | 0.745 | 8.5 | 6.6 | 25470.4 |
| **总体** | 141 | 0.872 | 0.064 | 0.936 | 0.007 | 0.901 | 0.861 | 0.780 | 0.655 | 0.857 | 7.3 | 5.0 | 19211.1 |

总体准确率 0.872，95% CI [0.8156, 0.9222]
