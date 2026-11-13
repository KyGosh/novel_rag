# p5_abl_person_s0

- 时间：2026-11-13T19:03:13；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": ["Person"], "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 122, 'limit_tokens': 14, 'limit_tool_calls': 5}；判分方式：{'rule': 83, 'judge': 56, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 11756.3, 'p95': 43090.1}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.776 | 1.000 | 5.8 | 2.6 | 13789.9 |
| 人物关系 | 33 | 0.970 | 0.000 | 0.970 | 0.000 | 0.970 | 0.970 | 0.970 | 0.707 | 0.970 | 7.4 | 4.8 | 18233.2 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.045 | 0.776 | 0.697 | 0.557 | 0.623 | 0.697 | 7.6 | 5.1 | 22040.9 |
| 跨章节情节 | 34 | 0.647 | 0.324 | 0.971 | 0.000 | 0.774 | 0.730 | 0.686 | 0.549 | 0.716 | 8.5 | 6.7 | 27204.6 |
| **总体** | 141 | 0.865 | 0.078 | 0.943 | 0.014 | 0.869 | 0.833 | 0.779 | 0.658 | 0.830 | 7.4 | 4.9 | 20639.4 |

总体准确率 0.865，95% CI [0.8014, 0.9149]
