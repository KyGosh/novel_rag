# p5_abl_think_s0

- 时间：2026-11-13T19:31:10；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": true, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 115, 'limit_tokens': 22, 'finish_implicit': 1, 'limit_tool_calls': 3}；判分方式：{'rule': 85, 'judge': 55, 'abstain': 1}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 41626.7, 'p95': 75911.1}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.786 | 1.000 | 6.8 | 3.3 | 17613.2 |
| 人物关系 | 33 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.717 | 1.000 | 8.5 | 5.8 | 23850.9 |
| 多跳 | 44 | 0.909 | 0.000 | 0.909 | 0.023 | 0.830 | 0.761 | 0.614 | 0.670 | 0.750 | 8.8 | 6.1 | 28712.5 |
| 跨章节情节 | 34 | 0.647 | 0.324 | 0.971 | 0.000 | 0.696 | 0.642 | 0.598 | 0.505 | 0.642 | 9.1 | 7.8 | 30595.3 |
| **总体** | 141 | 0.886 | 0.078 | 0.965 | 0.007 | 0.874 | 0.839 | 0.782 | 0.666 | 0.836 | 8.4 | 5.8 | 25667.1 |

总体准确率 0.886，95% CI [0.8298, 0.9362]
