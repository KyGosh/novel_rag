# p5_abl_nograph_s1

- 时间：2026-11-13T18:32:59；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 1；并发 8
- 终止原因：{'finish': 126, 'limit_tokens': 14, 'finish_implicit': 1}；判分方式：{'rule': 89, 'judge': 50, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25787.9, 'p95': 47660.0}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 0.967 | 0.933 | 0.933 | 0.704 | 0.933 | 5.7 | 2.4 | 12248.0 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.000 | 1.000 | 1.000 | 0.970 | 0.667 | 1.000 | 6.1 | 3.1 | 15082.3 |
| 多跳 | 44 | 0.886 | 0.000 | 0.886 | 0.045 | 0.773 | 0.670 | 0.511 | 0.523 | 0.670 | 7.4 | 4.4 | 19820.2 |
| 跨章节情节 | 34 | 0.676 | 0.235 | 0.912 | 0.000 | 0.814 | 0.745 | 0.657 | 0.553 | 0.745 | 8.2 | 5.6 | 24867.6 |
| **总体** | 141 | 0.865 | 0.057 | 0.922 | 0.014 | 0.877 | 0.822 | 0.744 | 0.604 | 0.822 | 6.9 | 4.0 | 18317.3 |

总体准确率 0.865，95% CI [0.8085, 0.922]
