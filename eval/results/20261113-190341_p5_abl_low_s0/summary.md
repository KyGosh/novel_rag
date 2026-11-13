# p5_abl_low_s0

- 时间：2026-11-13T19:09:32；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 3, "max_tool_calls": 6, "max_tokens": 20000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'limit_rounds': 24, 'finish': 67, 'limit_tokens': 45, 'limit_tool_calls': 5}；判分方式：{'rule': 83, 'judge': 54, 'abstain': 4}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 18698.9, 'p95': 26166.4}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.033 | 0.933 | 0.933 | 0.933 | 0.734 | 0.933 | 5.9 | 2.8 | 14145.3 |
| 人物关系 | 33 | 0.970 | 0.000 | 0.970 | 0.000 | 0.939 | 0.879 | 0.879 | 0.651 | 0.879 | 6.4 | 3.8 | 14356.1 |
| 多跳 | 44 | 0.795 | 0.023 | 0.818 | 0.068 | 0.754 | 0.659 | 0.481 | 0.585 | 0.659 | 6.2 | 4.1 | 16340.9 |
| 跨章节情节 | 34 | 0.500 | 0.353 | 0.853 | 0.000 | 0.696 | 0.627 | 0.613 | 0.505 | 0.627 | 6.3 | 4.6 | 17024.7 |
| **总体** | 141 | 0.801 | 0.092 | 0.894 | 0.028 | 0.822 | 0.761 | 0.702 | 0.613 | 0.761 | 6.2 | 3.9 | 15574.1 |

总体准确率 0.801，95% CI [0.7305, 0.8652]
