# p4_agent_dev_s1

- 时间：2026-11-12T01:15:06；commit `13ab7572ef`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 1；并发 4
- 终止原因：{'finish': 118, 'limit_tokens': 19, 'limit_rounds': 1, 'limit_tool_calls': 3}；判分方式：{'rule': 85, 'judge': 53, 'abstain': 3}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 14913.6, 'p95': 27657.3}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 0.967 | 0.967 | 0.967 | 0.748 | 0.967 | 5.9 | 2.6 | 13707.9 |
| 人物关系 | 33 | 0.970 | 0.000 | 0.970 | 0.000 | 0.970 | 0.970 | 0.970 | 0.641 | 0.970 | 7.3 | 4.7 | 18407.9 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.045 | 0.765 | 0.689 | 0.542 | 0.624 | 0.689 | 7.9 | 5.5 | 23108.4 |
| 跨章节情节 | 34 | 0.647 | 0.206 | 0.853 | 0.029 | 0.652 | 0.642 | 0.613 | 0.502 | 0.623 | 8.9 | 7.1 | 28092.5 |
| **总体** | 141 | 0.858 | 0.050 | 0.908 | 0.021 | 0.829 | 0.803 | 0.749 | 0.626 | 0.798 | 7.6 | 5.1 | 21210.0 |

总体准确率 0.858，95% CI [0.7943, 0.9149]
