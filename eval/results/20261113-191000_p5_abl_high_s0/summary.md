# p5_abl_high_s0

- 时间：2026-11-13T19:17:40；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 10, "max_tool_calls": 20, "max_tokens": 60000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 129, 'limit_tokens': 11, 'finish_implicit': 1}；判分方式：{'rule': 85, 'judge': 54, 'abstain': 2}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 21344.7, 'p95': 54523.0}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.825 | 1.000 | 5.9 | 2.8 | 13652.8 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.000 | 0.939 | 0.909 | 0.849 | 0.624 | 0.909 | 7.3 | 4.7 | 17944.8 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.045 | 0.780 | 0.670 | 0.492 | 0.605 | 0.670 | 7.8 | 5.3 | 23495.5 |
| 跨章节情节 | 34 | 0.676 | 0.294 | 0.971 | 0.000 | 0.770 | 0.711 | 0.696 | 0.556 | 0.667 | 9.4 | 7.8 | 31757.7 |
| **总体** | 141 | 0.865 | 0.071 | 0.936 | 0.014 | 0.862 | 0.806 | 0.733 | 0.645 | 0.795 | 7.7 | 5.2 | 22094.5 |

总体准确率 0.865，95% CI [0.8014, 0.922]
