# p4_lat2_agent

- 时间：2026-11-12T18:18:00；commit `839f2be161`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，28 题；采样编号 0；并发 1
- 终止原因：{'finish': 24, 'limit_rounds': 1, 'limit_tokens': 3}；判分方式：{'rule': 16, 'judge': 12}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 16576.1, 'p95': 28706.3}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 7 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.691 | 1.000 | 7.4 | 4.4 | 19048.0 |
| 人物关系 | 7 | 0.714 | 0.143 | 0.857 | 0.000 | 1.000 | 1.000 | 0.857 | 0.643 | 1.000 | 7.3 | 5.0 | 18249.0 |
| 多跳 | 7 | 1.000 | 0.000 | 1.000 | 0.000 | 0.833 | 0.762 | 0.571 | 0.476 | 0.762 | 7.7 | 5.9 | 22772.9 |
| 跨章节情节 | 7 | 0.714 | 0.143 | 0.857 | 0.000 | 0.857 | 0.857 | 0.857 | 0.714 | 0.857 | 8.0 | 6.3 | 23899.0 |
| **总体** | 28 | 0.857 | 0.071 | 0.929 | 0.000 | 0.923 | 0.905 | 0.821 | 0.631 | 0.905 | 7.6 | 5.4 | 20992.2 |

总体准确率 0.857，95% CI [0.7143, 0.9643]
