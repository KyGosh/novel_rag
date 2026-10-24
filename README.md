# novel_rag
GraphRAG focus on novel situation

我们要创建一个基于混合检索的小说智能问答系统，此系统需面向长篇小说跨章节问答， 构建基于 LangGraph 的 GraphRAG 系统， 解决人物关系复杂、 关键情节分散及传统 RAG 对隐含关系召回不足的问题。
技术栈： LangGraph、 GraphRAG、 Milvus、 Elasticsearch、 Neo4j、 Python、 FastAPI
技术亮点：
混合检索与融合： 实现 Elasticsearch BM25 与 Milvus 向量双路召回， 采用 RRF 融合候选结果， 并接入 Reranker 二次排序， 提升人物名、 专有名词及复杂语义问题的召回效果。
知识图谱增强与多跳检索： 从小说中抽取人物、 组织、 地点、 事件及其关系， 构建 Neo4j 知识图谱； 结合实体识别、实体链接和图谱查询， 将分散于不同章节的信息关联起来， 支持人物关系追踪、 事件因果关联及跨章节复杂问题分析。
RAG 评测与性能优化： 构建覆盖事实查询、 人物关系、 多跳推理和跨章节情节理解的测试集， 使用 Recall@K、 MRR等指标评估检索质量， 并结合 Ragas 评估答案相关性、 忠实度等生成质量指标。

## 快速开始（开发环境）

```bash
cp .env.example .env              # 填写 NEO4J__PASSWORD、LLM__API_KEY
docker compose up -d elasticsearch neo4j milvus
uv sync                           # torch 从 PyTorch cu130 索引安装
uv run pytest                     # 单测（Fake 实现，不依赖外部服务）
uv run python scripts/check_env.py   # 环境冒烟检查
uv run pytest -m integration      # 集成测试（需要 docker、GPU、Kimi key）
```
