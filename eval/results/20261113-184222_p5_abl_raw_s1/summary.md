# p5_abl_raw_s1

- 时间：2026-11-13T18:49:02；commit `4b884b98fe`
- 系统：{"name": "agent", "tools": ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"], "limits": {"max_rounds": 6, "max_tool_calls": 12, "max_tokens": 40000, "max_calls_per_round": 4, "max_error_rounds": 3}, "thinking": false, "act_memory": "raw", "graph_scope": null, "models": {"agent": "workbuddy/deepseek-v4.1-flash", "reflect": "workbuddy/deepseek-v4.1-flash", "answer": "workbuddy/deepseek-v4.1-flash"}, "prompts": ["agent_plan.v1", "agent_act_raw.v1", "agent_reflect.v1", "qa_answer.v1"], "retriever": {"name": "bm25+vector+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}}, "route_top_k": {"bm25": 50, "vector": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}}
- 评测集：asoiaf_v1，划分 dev，141 题；采样编号 1；并发 8
- 终止原因：{'finish': 115, 'limit_tokens': 25, 'finish_implicit': 1}；判分方式：{'rule': 83, 'judge': 51, 'abstain': 7}；无效引用 0 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 21157.8, 'p95': 34374.3}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | note_recall | cite_recall | cite_precision | ctx_recall | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 0.933 | 0.933 | 0.933 | 0.765 | 0.933 | 5.3 | 2.3 | 16847.0 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.000 | 1.000 | 1.000 | 0.970 | 0.642 | 1.000 | 6.4 | 3.8 | 21066.1 |
| 多跳 | 44 | 0.864 | 0.000 | 0.864 | 0.091 | 0.769 | 0.724 | 0.572 | 0.652 | 0.724 | 7.0 | 4.5 | 27482.2 |
| 跨章节情节 | 34 | 0.706 | 0.176 | 0.882 | 0.088 | 0.765 | 0.686 | 0.613 | 0.526 | 0.686 | 7.5 | 5.6 | 30319.8 |
| **总体** | 141 | 0.865 | 0.043 | 0.908 | 0.050 | 0.857 | 0.824 | 0.752 | 0.646 | 0.824 | 6.6 | 4.2 | 24402.0 |

总体准确率 0.865，95% CI [0.8014, 0.922]
