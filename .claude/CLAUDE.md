# CLAUDE.md — novel_rag

通用的长篇小说跨章节问答 GraphRAG 系统：LangGraph 编排，ES BM25 + Milvus 向量混合检索，Neo4j 知识图谱增强。演示和评测数据集是《冰与火之歌》中文版。

- 进度与决策：`docs/progress.md`；各阶段方案：`docs/phases/Px.md`；各模块的详细设计要点：`docs/design-notes.md`（做对应阶段方案时查阅）。
- 最高原则：每一项优化都要有可复现的评测数字。简历上的数字只能来自 `eval/results/`。

## 协作约定

- 按阶段 P0→P7 推进，每个阶段开一个新会话，结束时更新 `docs/progress.md`。
- 计划和方案阶段只讲思路（目标、步骤、取舍、验收标准），不写详细代码。
- subagent 只用于查资料、广泛搜索和代码审查。

## 技术栈（版本已锁定，改动要记入 progress.md）

Python 3.11 + uv；ES 8.19.23 + IK 8.19.23（两者必须完全一致）；Milvus 2.6.25 standalone；Neo4j 5.26 LTS + APOC；LLM 用 Kimi（模型名放在 config，按任务分别配置）；bge-m3 和 bge-reranker-v2-m3 跑在本地 GPU（RTX 4060 Ti 8GB）。

## 硬性约束

1. **通用性**：代码里不写死任何书籍专属知识。书籍相关信息（目录、别名、POV、schema 扩展）都是 `data/books/{book_id}/` 下的可选配置，没有这些配置也必须能跑通。支持 txt、md、epub 和有文字层的 pdf，各格式解析后统一成同一种中间表示，下游步骤不感知格式。
2. **抽象层**：LLM、Embedding、Reranker、视觉模型只能经过 `llm/` 调用，业务代码不能直接 import 厂商 SDK。所有参数（top_k、阈值、模型名）都集中放在 `config.py`。
3. **ID 与坐标**：
   - `chunk_id = {book_id}:{chapter_idx:04d}:{seq:04d}`，确定性生成，ES、Milvus、Neo4j 三处共用。
   - 字符位置以规范化后的 `text.txt` 为准，不要对全文做 NFKC 规范化。
   - 实体 ID 为系列级。
4. **防剧透**：`max_reading_order` 是一等参数，必须覆盖 ES、Milvus、Neo4j 和摘要索引所有检索路径，并配有单测。
5. **证据**：知识图谱中的实体和关系必须带来源 chunk 和 reading_order，没有证据就不入库。关系会随剧情变化，不能互相覆盖。
6. **评测**：
   - 金标证据用字符区间标注，不存 chunk_id。
   - 分 dev / test 集，调参只看 dev。
   - 按题型报告指标。
   - 每次实验都要记录 commit、配置快照和逐题结果。
7. **安全**：Cypher 以模板为主。Text2Cypher 只允许参数化的只读查询，并强制 LIMIT。输入文件按不可信数据处理。
8. **版权**：原文只在本地处理。`data/` 不入库；评测集、README 和截图中不能出现大段原文。

## 工作流

- 改动检索、图谱或 prompt 后，提交前要跑评测，与基线对比。
- 核心逻辑必须有单测，测试用自己构造的样例，不用版权原文。外部服务用 mock 替代，或者放到 `tests/integration`。
- prompt 以文件形式存放并带版本号；LLM 调用结果按内容 hash 缓存。
- 不提交 `data/`、`.env`、模型权重和 `.idea/`。
