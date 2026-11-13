# p5_abl_nogq_s0

- 时间：2026-11-13T18:57:56；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 125, 'limit_tokens': 11, 'limit_rounds': 2, 'finish_implicit': 2, 'limit_tool_calls': 1}；判分方式：{'rule': 87, 'judge': 53, 'abstain': 1}；无效引用 3 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25521.7, 'p95': 45199.7}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 0.967 | 0.967 | 0.967 | 0.783 | 0.967 | 5.7 | 2.5 | 12147.2 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.000 | 1.000 | 1.000 | 0.970 | 0.654 | 1.000 | 6.8 | 3.9 | 15400.9 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.023 | 0.769 | 0.727 | 0.591 | 0.633 | 0.727 | 7.5 | 4.8 | 19530.0 |
| 跨章节情节 | 34 | 0.647 | 0.176 | 0.824 | 0.000 | 0.662 | 0.588 | 0.544 | 0.464 | 0.574 | 9.2 | 7.1 | 26951.1 |
| **总体** | 141 | 0.858 | 0.043 | 0.901 | 0.007 | 0.839 | 0.808 | 0.748 | 0.630 | 0.805 | 7.4 | 4.6 | 18782.3 |

总体准确率 0.858，95% CI [0.7943, 0.9149]
