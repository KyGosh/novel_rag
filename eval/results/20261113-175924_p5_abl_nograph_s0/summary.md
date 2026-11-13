# p5_abl_nograph_s0

- 时间：2026-11-13T18:07:58；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 123, 'limit_tool_calls': 1, 'limit_tokens': 16, 'finish_implicit': 1}；判分方式：{'rule': 89, 'judge': 50, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 24649.5, 'p95': 50110.5}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 0.967 | 0.967 | 0.967 | 0.776 | 0.967 | 5.6 | 2.4 | 11996.2 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.000 | 1.000 | 0.939 | 0.939 | 0.669 | 0.939 | 5.9 | 3.1 | 14956.4 |
| 多跳 | 44 | 0.932 | 0.000 | 0.932 | 0.045 | 0.845 | 0.773 | 0.549 | 0.637 | 0.773 | 7.3 | 4.4 | 19812.3 |
| 跨章节情节 | 34 | 0.618 | 0.294 | 0.912 | 0.000 | 0.774 | 0.696 | 0.657 | 0.513 | 0.681 | 8.3 | 5.7 | 23713.4 |
| **总体** | 141 | 0.872 | 0.071 | 0.943 | 0.014 | 0.890 | 0.835 | 0.755 | 0.644 | 0.831 | 6.9 | 4.0 | 17953.5 |

总体准确率 0.872，95% CI [0.8156, 0.922]
