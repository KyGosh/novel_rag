# p4_agent_dev_s0

- 时间：2026-11-12T01:04:42；commit `13ab7572ef`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 4
- 终止原因：{'finish': 128, 'limit_rounds': 2, 'limit_tokens': 9, 'limit_tool_calls': 2}；判分方式：{'rule': 86, 'judge': 52, 'abstain': 3}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 13528.7, 'p95': 29366.0}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.967 | 0.967 | 0.806 | 0.967 | 5.9 | 2.7 | 13974.4 |
| 人物关系 | 33 | 0.970 | 0.030 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.654 | 1.000 | 7.2 | 4.6 | 17847.3 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.068 | 0.765 | 0.708 | 0.526 | 0.618 | 0.708 | 7.5 | 5.0 | 21396.0 |
| 跨章节情节 | 34 | 0.676 | 0.235 | 0.912 | 0.000 | 0.765 | 0.701 | 0.657 | 0.522 | 0.686 | 8.8 | 6.9 | 27610.6 |
| **总体** | 141 | 0.872 | 0.064 | 0.936 | 0.021 | 0.870 | 0.830 | 0.762 | 0.644 | 0.826 | 7.4 | 4.9 | 20485.0 |

总体准确率 0.872，95% CI [0.8156, 0.922]
