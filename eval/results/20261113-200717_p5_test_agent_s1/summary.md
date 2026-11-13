# p5_test_agent_s1

- 时间：2026-11-13T20:15:39；commit `3fcb8a787e`（有未提交改动）
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 test，141 题；采样编号 1；并发 8
- 终止原因：{'finish': 133, 'limit_tokens': 8}；判分方式：{'rule': 89, 'judge': 50, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25942.4, 'p95': 43535.9}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.033 | 1.000 | 0.000 | 1.000 | 0.983 | 0.967 | 0.778 | 0.983 | 6.1 | 3.0 | 13984.5 |
| 人物关系 | 33 | 0.939 | 0.030 | 0.970 | 0.030 | 0.803 | 0.788 | 0.773 | 0.568 | 0.788 | 7.2 | 4.4 | 18207.8 |
| 多跳 | 44 | 0.909 | 0.045 | 0.955 | 0.023 | 0.739 | 0.636 | 0.481 | 0.605 | 0.636 | 7.9 | 5.5 | 23052.7 |
| 跨章节情节 | 34 | 0.706 | 0.294 | 1.000 | 0.000 | 0.804 | 0.721 | 0.681 | 0.576 | 0.706 | 8.1 | 5.8 | 23965.0 |
| **总体** | 141 | 0.879 | 0.099 | 0.979 | 0.014 | 0.825 | 0.766 | 0.701 | 0.626 | 0.762 | 7.4 | 4.8 | 20209.4 |

总体准确率 0.879，95% CI [0.8227, 0.9291]
