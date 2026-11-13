# p5_abl_nosum_s0

- 时间：2026-11-13T18:17:06；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "notes", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 0；并发 8
- 终止原因：{'finish': 125, 'limit_tokens': 16}；判分方式：{'rule': 84, 'judge': 54, 'abstain': 3}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 25531.8, 'p95': 45416.7}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 0.000 | 1.000 | 0.000 | 0.967 | 0.967 | 0.967 | 0.717 | 0.967 | 5.8 | 2.8 | 13289.1 |
| 人物关系 | 33 | 0.970 | 0.000 | 0.970 | 0.000 | 0.939 | 0.939 | 0.939 | 0.697 | 0.939 | 7.3 | 4.6 | 16589.3 |
| 多跳 | 44 | 0.886 | 0.000 | 0.886 | 0.068 | 0.758 | 0.610 | 0.473 | 0.583 | 0.610 | 7.7 | 5.4 | 22056.2 |
| 跨章节情节 | 34 | 0.735 | 0.206 | 0.941 | 0.000 | 0.799 | 0.745 | 0.701 | 0.550 | 0.730 | 8.2 | 6.4 | 24902.5 |
| **总体** | 141 | 0.894 | 0.050 | 0.943 | 0.021 | 0.855 | 0.795 | 0.742 | 0.631 | 0.792 | 7.3 | 4.9 | 19597.7 |

总体准确率 0.894，95% CI [0.8369, 0.9433]
