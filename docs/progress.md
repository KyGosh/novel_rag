# 进度与决策记录

新会话开始时先读本文件、`.claude/CLAUDE.md` 和当前阶段的 `docs/phases/Px.md`；需要模块细节时查 `docs/design-notes.md`。每个阶段结束时更新"当前状态"和"决策记录"。

## 当前状态

- 阶段：**P2 已完成（2026-11-06，分支 `p2-hybrid`）**，下一阶段 P3（知识图谱 + 章节摘要 + 并行做评测集 v1）。P2 的六条验收标准全部达成：
  - **卷二至卷五 `text.txt` 已定稿（2026-11-06，用户审阅抽查报告后锁定）**，sha256：卷二 `6f9c2a0fa3d39b3cbcfbc9bd0aecf78c3d6533f9992e616bfab816564cfb4b36`、卷三 `5e1f976f8f953ed17f24bd79e0eaa518e4169dfc92b12b66019e08a5e25caef3`、卷四 `6930005452dacf122323133788b16eea6b072b50234595983f09c341ad1747b9`、卷五 `8205202c5e56e7cbf45483b2da74e3c1eee9a3ca9a759e8a9d104fcdcd720509`。reading_order：卷一 0–73、卷二 74–144、卷三 145–227、卷四 228–274、卷五 275–348
  - **全五卷已分块并索引**：chunk 版本 `0c94b7e4`，卷一至卷五 1181 / 1271 / 1577 / 1163 / 1584，共 6776 个；ES 索引 `chunks_0c94b7e4_es1` 与 Milvus collection `chunks_bgem3_1024_0c94b7e4` 的数量、ID、字段完全一致（`scripts/check_index.py`）
  - **混合检索链路**：ES BM25（IK；可选 CJK bigram 子字段）+ Milvus 向量 → RRF → bge-reranker → 可选 small-to-big → 按 token 预算装箱。`max_reading_order` 在 BM25、向量、融合、rerank、扩展每一步都过滤并复查，有单测（ES / Milvus / 内存三种过滤语义一致）和集成测试
  - **消融表 v1**：`eval/results/ablation_v1.md`（规格 `eval/ablations/v1.json`），16 行，每行带总体与按题型指标、95% bootstrap 置信区间、对照行配对差值、延迟
  - **固定流程 RAG 基线**已写入 config 默认值（见决策记录"P2 结论"），test 只跑一次
  - txt 解析器与格式无关性测试（全五卷 epub → txt 往返逐字一致）
- 最新评测数字（P2 固定流程 RAG 基线，设置 A：检索范围卷一，不开防剧透过滤）：
  - dev（`eval/results/20261106-200113_p2a_rr40`）：总体 Recall@5 0.561 / @20 0.739，MRR 0.565，cov@4000 0.661；事实 R@5 1.000、关系 0.467、多跳 0.633、跨章节 0.144
  - test（`eval/results/20261106-201258_p2_baseline_test`）：总体 Recall@5 0.639 / @20 0.787，Hit@20 0.950，MRR 0.682，cov@4000 0.714；事实 0.933、关系 0.744、多跳 0.633、跨章节 0.244。相比 P1 纯向量 test：R@5 0.422 → 0.639，MRR 0.384 → 0.682
  - 检索延迟 P50 约 690 ms（其中 rerank 约 630 ms，BM25 7 ms，向量 57 ms）
  - 结论：跨章节情节题仍是最大短板（dev R@5 0.144），固定流程检索拿不到分散在多章的证据，这是 P3 章节摘要和 P4 Agent 要解决的问题
- P1 状态（2026-11-06 完成），以下为 P1 交接内容
  - 第一步已实现：`scripts/ingest.py`（epub 解析、规范化、卷与章节识别、注释剥离、质检报告、定稿保护），全五卷质检通过（349 章，与目录一致；27 条注释全部配对），人工抽查通过
  - **卷一 `text.txt` 已定稿（2026-10-25）**：sha256 `202f6f3a1f81035d9980812a4276a36ade7b69ac711f825316f9b59337ef4780`，543,372 字，74 章（序幕 0 + 正文 72 + 附录 73）；原文 epub sha256 `48e924072b5fdf2c4b4a763ddda657c3cb641b42bd50b63637cd1b6a5883ac44`，parser `epub-1`，规则 `norm-1`。重新导入时 hash 不同会被 `text.lock` 拒绝
  - **评测集 asoiaf_v0 已冻结（2026-11-06，commit e3f9bde）**：四类各 30 题，dev / test 各 60 题，证据覆盖卷一 71 章（225 段，204 段互不相同）。195 道候选题全部校验：通过 184（其中修改 13）、丢弃 11，通过率 94.4%；跨章节题通过率最低（77.3%）。校验方式：Claude 出题并逐题预审给建议，用户对照原文证据逐批确认（已写入 manifest，非完全独立的人工校验）。暂无人工题子集，留到 v1。未入选的 64 道通过题在 `reserve.jsonl`
  - A 线已完成并跑通全链路：切句 + 分块（`scripts/chunk.py`，卷一 chunk 版本 `0c94b7e4`：1181 个，字数 P50 517 / P99 644，reranker token P99 511 + 查询 64 ≤ 1024）→ Milvus 导入（`scripts/index_milvus.py`，collection `chunks_bgem3_1024_0c94b7e4`，幂等、向量按文本 hash 缓存）→ 向量检索 → 评测运行器（`scripts/run_eval.py`）→ 最简生成（`scripts/ask.py`、`--generate`）
  - **顺延项**（P1 范围内未做，都不影响已定稿的 `text.txt` 和评测集；2026-11-06 目标调整后，插图和 pdf 已暂停，txt 解析器与分块消融并入 P2）：
    - 插图提取与分类：顺延到 P2 开头（P5 插图理解的前置）
    - txt / md 解析器与格式无关性测试：顺延到 P2 开头（硬性约束 1 要求支持，测试用自拟样例）
    - pdf 解析器：按 P1 方案允许顺延，放到 P5 前
    - 可选消融（分块 300 / 500 / 800、标题前缀）：并入 P2 的消融表 v1
    - 人工题子集：评测集 v1 时补，用于衡量出题措辞贴近原文带来的偏差
  - 新会话开工前：`docker compose up -d` 启动容器（关机后会停止，数据在命名卷里）；`.env` 中设置 `LLM__MAX_RPM=3`
- P1 纯向量 baseline 数字（bge-m3 + Milvus HNSW，chunk `0c94b7e4`，不开防剧透过滤）：
  - dev（`eval/results/20261106-105947_baseline_vector_dev`）：总体 Recall@5 0.406 / @20 0.644，Hit@20 0.817，MRR 0.403；事实 R@5 0.800、关系 0.367、多跳 0.333、跨章节 0.122
  - test（`eval/results/20261106-110025_baseline_vector_test`）：总体 Recall@5 0.422 / @20 0.671，Hit@20 0.833，MRR 0.384；事实 0.667、关系 0.567、多跳 0.233、跨章节 0.222
  - 检索延迟 P50 约 62–66 ms。结论：纯向量对事实题够用，多跳与跨章节题明显不足（P2 的 BM25 / rerank / small-to-big、P4 的章节摘要要解决的问题）

## 开发阶段

（2026-11-06 按 Agent 目标重排，见决策记录"项目目标调整"）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | docker-compose（ES+IK / Milvus / Neo4j）、uv 项目骨架、config、`llm/` 抽象层、冒烟测试 | 已完成 |
| P1 | epub 解析与规范化、卷一 `text.txt` 定稿、分块、Milvus 向量 baseline、最简生成、评测集 v0、Recall / MRR | 已完成 |
| P2 | ES BM25 + RRF + rerank + small-to-big，固定流程 RAG 的消融表 v1（含分块参数、注释是否入索引），确定固定 RAG 的基线配置；txt 解析器与格式无关性测试；**全五卷导入与索引** | 已完成 |
| P3 | 知识图谱（实体关系抽取、同名不同人的消歧、Neo4j、图谱查询模板、回灌 IK 词典）+ 章节摘要；**同时并行做评测集 v1**（三类主题型每类 ≥100 题 + 事实对照组 + 人工题子集） | 未开始 |
| P4 | 工具层 + LangGraph Agent 循环（规划 → 检索 → 反思 → 作答）+ 证据笔记 / 章节摘要分层记忆 + 引用校验 + 上限与重试；`llm/` 支持 function calling；Agent 评测框架（答案准确率、judge 一致率、证据召回、调用次数、延迟）；第一轮对比：闭卷 / 固定 RAG / Agent | 未开始 |
| P5 | 全五卷语料上的完整对比与 Agent 消融（去掉图谱工具、去掉笔记、去掉摘要、不同轮数上限），失败模式分析 | 未开始 |
| P6 | 性能（缓存、并行工具调用、延迟拆解）+ 可选演示 + README 与简历材料 | 未开始 |

## 决策记录

- 按阶段 P0→P7 推进，模块分阶段逐步实现，不按模块顺序开发。
- 每个阶段开一个新会话，靠本文件交接；subagent 只用于查资料和代码审查。
- 计划和方案阶段只讲思路，不写详细代码。
- 目标书籍为《冰与火之歌》中文版全五卷，P1–P3 先只用卷一。原文受版权保护，只在本地处理，不入库。
- LLM 选用 Kimi（Moonshot，OpenAI 兼容接口）。（2026-11-06 已改为 DeepSeek，见下文"LLM 切换为 DeepSeek"；下面几条提到 Kimi 的记录保留为历史）
- Embedding（bge-m3）和 Reranker（bge-reranker-v2-m3）在本地 GPU 上运行。
- 版本：ES 8.19.23 + IK 8.19.23、Milvus 2.6.25、Neo4j 5.26.32-community（LTS），Python 3.11。
- 阅读进度和防剧透统一用全局 `reading_order` / `max_reading_order`（按卷序 → 章序编号）。
- 规范化全文 `text.txt` 必须先定稿，再标注评测集（金标按"卷 + 章 + 字符区间"标注）。
- 章节摘要放在 P4，与"情节类"路由一起做。

- 系统是通用的：书籍专属信息（目录、别名表、POV 映射、schema 扩展）都是可选配置，没有配置也能跑通。原文的 sha256 由导入流程自动记录，不需要人工提供版本信息。
- 输入支持 txt / md / epub / pdf（仅限有文字层的 pdf；扫描版暂不支持，OCR 留作以后的插件）。各格式解析后统一成同一种中间表示，下游步骤不感知格式。
- 插图分两步处理：P1 只做提取和过滤（锚定到正文位置）；用视觉模型生成描述并入库是 P5 的可选增强，届时再决定用 Kimi 的视觉模型还是本地模型。
- Kimi 的模型名不写死，在 config 中按任务（抽取、生成、judge）分别配置。
- 硬件：RTX 4060 Ti 8GB（驱动支持 CUDA 13.3）、内存 32GB、Ryzen 7 5700X。
- 部署位置：ES、Milvus、Neo4j 跑在 Docker Desktop 容器里（WSL2 后端，数据用命名卷）；Python 应用和模型在开发期（P0–P5）跑在 Windows 原生环境，通过 localhost 端口访问容器；到 P6 再把应用打包成 Linux 镜像，用 compose 统一编排。代码保持跨平台。WSL 的内存上限设为 12GB，其余内存留给 Windows 端的模型和 IDE。

- P0 实际锁定的依赖（2026-10-22）：torch 2.14.1+cu130（PyTorch cu130 索引，pyproject 中 explicit 索引指定）、FlagEmbedding 1.4.2、transformers 5.20.2、openai 3.36.0（HTTP 层已改为 httpx2）、elasticsearch 8.19.3、pymilvus 2.6.17、neo4j 驱动 6.4.1、pydantic 2.14 / pydantic-settings 2.16、diskcache 5.6.3（LLM 缓存）、tenacity 9.2。精确版本以 `uv.lock` 为准。
- Milvus 的 etcd（v3.5.25）、minio（RELEASE.2024-12-18T13-15-44Z）版本照抄官方 v2.6.25 standalone compose；MQ 用官方默认的 woodpecker。minio 不映射宿主机端口。
- IK 从 `get.infini.cloud/elasticsearch/analysis-ik/8.19.23` 安装；`ext_dict` 改为 `custom/novel_ext.dic`，`custom/` bind mount 到 `docker/elasticsearch/ik-dict/`。IK 只在启动时加载本地词典，改词典后要重启 ES 并重建索引。
- LLM 按任务配置（`llm.tasks.{extraction,generation,judge,smoke}`），默认 `kimi-k2.6` 并关闭思考（`extra_body.thinking.type=disabled`）；环境变量可只覆盖某任务的某字段。结构化输出默认用 Kimi 的 `response_format=json_schema`，可切换 `json_object`（schema 写进 system prompt）；校验失败带纠错反馈重试。
- 缓存键 = sha256(模型及请求参数, 消息, schema, prompt 版本)；只缓存成功结果；缓存命中计入调用次数但不计 token。
- P0 环境检查（2026-10-22）：`scripts/check_env.py` 7 项全部通过（ES / Milvus / Neo4j / GPU / Embedding / Reranker / Kimi）。`.wslconfig`：memory=14GB、swap=4GB、`kernelCommandLine` 持久化 `vm.max_map_count=262144`（已验证生效）。空载内存：ES 1.5GB、Neo4j 1.5GB、Milvus 三件套约 0.25GB。
- **IK 默认词典基线**（自拟测试句）：`ik_max_word` 与 `ik_smart` 都把"琼恩·雪诺"切成"琼 / 恩 / 雪 / 诺"四个单字，"守夜人""长城"等常用词正常成词。这是 P3 回灌人名词典的依据。已验证回灌流程：往 `novel_ext.dic` 加词 → 重启 ES → 该词作为完整 token 输出。
- 本地模型权重放在项目 `models/`（gitignore）。`paths.resolve_model()` 优先从 `models/{名称最后一段}` 加载，没有才走 HuggingFace Hub；配置和实验记录里的模型名仍是 `BAAI/bge-m3` 等 Hub 名。bge-m3 的 `onnx/`（2.2GB）不需要。
- uv 缓存放在 `D:/Project/novel_rag/.uv-cache`，由用户级 `%APPDATA%\uv\uv.toml` 的 `cache-dir` 指定，与 `.venv` 同盘以便硬链接。pyproject 里写相对路径会按当前工作目录解析，不可用。换机器时不需要这一步。
- 厂商 SDK（openai / FlagEmbedding / torch / transformers）只允许在 `llm/` 中 import，由 `tests/unit/test_import_boundaries.py` 强制。

- P1 第十二节决策（2026-10-24）：人工校验可以投入；出题先试 10 题、看过花费再全量；模糊匹配用 rapidfuzz；pdf 默认 pypdfium2；注释 P1 只提取保存，P2 用消融决定是否入索引。
- 导入产物布局：系列级可选配置 `data/books/{series_id}/book.toml`（卷名纠正、POV 映射、section_type 覆盖）；每卷产物 `data/books/{series_id}-{卷号}/`（`text.txt`、`chapters.jsonl`、`notes.jsonl`、`manifest.json`，定稿后多一个 `text.lock`）。`text.txt` 一卷一个、只含 narrative + appendix；章节是偏移表，不按章拆文件（章节切分改了不影响金标区间）。已定稿的卷若重新生成的 hash 不同，导入拒绝覆盖，除非 `--force`。
- 不用 ebooklib（AGPL），OPF / NCX / nav 自行解析；XHTML 先按 XML 严格解析（HTML 命名实体转数字引用，不加载 DTD），失败才退回 lxml 的 HTML 解析器。新增依赖 lxml、Pillow。
- 导入实测发现（修正了 P1.md 第一节的判断）：
  - "汉字?汉字"并不普遍是丢失的间隔号。卷二全程用半角 `?` 作问号（三个分册共 2711 处），间隔号正常；卷一只有注释里一处"曼斯?雷德"。规则改为按分册判定：不夹在汉字之间的 `?` 占比 ≥ `qmark_punct_ratio`（0.2）时视为半角问号，统一成全角"？"；否则才把汉字之间的 `?` 修复为 `·`。
  - 汉字之间的空格不再删除（只计数）：附录表格靠空格分列。
  - 卷三（上）（下）的目录锚点指向标题页中间的 `h1`。分册起点规则：锚点之前只有不到 50 字且没有标题时，前移到文件开头。
  - 卷一附录的末尾是一段没有标题的译后记，无法用通用规则拆出，随附录一起记为 appendix。
- 评测集出题方式（2026-10-25）：出题不调用 Kimi（Kimi 只作为被测系统的模型），由 Claude 在会话中按 `prompts/qgen_guidelines.v1.md` 出题；脚本负责采样窗口（每章一个、固定种子）、导出任务、导入并定位证据。可复现性靠候选题文件 hash + 规范版本号 + 冻结后的评测集文件保证，出题过程本身不能一键重跑。出题模型与被测模型不同，避免自我偏好；"人工题"子集用来衡量出题措辞贴近原文带来的偏差。新增依赖 rapidfuzz。
- 评测集格式：`eval/datasets/{name}/items.jsonl`（id、题型、问题、答案、证据区间、来源、划分、`max_reading_order`=最后一段证据所在章的 reading_order）+ `manifest.json`（定稿 text sha256、规范版本、候选文件 hash、定位与校验统计）+ `reserve.jsonl`（未入选的通过题）。加载时 text hash 不一致直接报错。冻结时检查问题和答案中不得出现原文连续 20 字。dev / test 在"题型 × 来源"分组内按 sha256(id) 交替划分。
- 第二轮人名共现配对的教训：按名字第一段配对会把同名不同人（如琼恩·雪诺与琼恩·艾林）配到一起，190 个组合任务中有 68 个出不了合格题。P3 实体消歧要处理同名问题。
- P1 结论（2026-11-06）：
  - 纯向量 baseline 在事实题上已经可用（dev R@20 = 1.0），多跳与跨章节题是主要短板（dev R@5 分别为 0.333 / 0.122）。dev / test 每类只有 15 题，按题型的数字波动大（如关系题 dev 0.367、test 0.567），结论以总体和趋势为准；P2 起扩充题量。
  - 生成抽查（dev 60 题，Kimi，结果只存 `data/`）：检索前 6 名命中证据时仍有 12 题回答"无法确定"，其中 8 道是多跳题——只检索到含答案的第二跳，没检索到识别桥接实体的第一跳，模型无法确认题目所指。多跳的瓶颈在检索，P2 的 BM25、small-to-big 与后续 query 改写要针对这一点。
  - 评测集出题教训：组合题强制"每个窗口都取证据"会凑入与答案无关的证据，拉低按证据统计的 Recall，跨章节题也因此丢弃最多（10/44）。下一版出题规范要改为"只取支撑答案的证据，但至少跨两章"。
  - 人名共现配对会把同名不同人配到一起（琼恩·雪诺 / 琼恩·艾林），P3 实体消歧要处理。
- A 线设计（2026-10-26）：
  - 切句：句末标点 + 后引号吞入；引号内不断句（超过 200 字的对话允许断开）；省略号只在后引号前或段尾算句末；引语后紧跟归属语（"他说"）时与归属语算同一句，避免切出孤立的短句。
  - 分块：只切章节正文，章节标题作为元数据（是否加入编码文本由 `embed_text_template` 消融决定）；整句贪心到 500 字、上限 800、重叠 ≤ 90 字整句、章末 < 150 字并入前块；不变量（切片一致、不跨章、全覆盖、确定性）在代码中检查。chunk 版本 = 参数 + 切句 / 分块规则版本的 hash。
  - small-to-big（命中块扩展到 `seq ± 1` 或父块）仍按原设计放在 P2，rerank 之后做，用"固定 token 预算下的证据覆盖率"证明收益；P1 baseline 只用单一粒度，避免大块虚高 Recall。
  - token 计数经 `llm/` 的 `TokenCounter`（bge-reranker tokenizer），transformers 仍只在 `llm/` 中 import。
  - 评测：相关性 = 证据区间与 chunk 重叠 ≥ min(区间 50%, 50 字)；Recall@K 按证据统计；默认只跑 dev，test 需显式 `--split test`；baseline 不开防剧透过滤（`--spoiler-filter` 作为单独实验）；结果目录不含原文。
  - 草稿评测集：`freeze_dataset.py --draft` 收录未校验题，输出到 `data/eval_work/{dataset}/draft/`，只用于跑通链路。
  - pymilvus 2.6 搜索结果中主键以字段名（`chunk_id`）返回，不是 `id`。
- P1 规划要点（2026-10-22）：一个文件里可以包含多卷，卷由目录顶层的分册合并而成（去掉"上/中/下"后缀后同名），可选配置可以纠正卷名；注释（`<sup>[n]</sup>` 加文件末尾的注释段落）从正文中剥离，单独存放；插图分为装饰、附录图示、插画/地图三类；全五卷都要解析，但索引和评测只用卷一；相关性判定按"证据区间与 chunk 的重叠字数 ≥ min(区间的 50%, 50 字)"。

- **LLM 切换为 DeepSeek（2026-11-06）**：经本地 WorkBuddy 代理（`http://127.0.0.1:3425/v1`，OpenAI 兼容，key 不校验但必须非空），模型 `workbuddy/deepseek-v4.1-flash`，各任务的默认值都已改成它。实测结论：
  - 关闭思考只认顶层 `reasoning_effort: "none"`；`reasoning.effort`、`thinking.type` 会被忽略。
  - `response_format=json_schema` 会被静默忽略（照常返回纯文本），所以 `structured_mode` 默认改为 `json_object`。
  - 支持 function calling（P4 的 Agent 可以直接用）。
  - 没有 Kimi 那样的 RPM 3 限制，`.env` 中已去掉 `LLM__MAX_RPM`。
  - `check_env.py` 的 kimi 检查项改名为 llm，已通过。
  - 缓存键包含模型名，以前 Kimi 的缓存不会被复用。P1 的检索 baseline 不涉及 LLM，不受影响；P1 的生成抽查是 Kimi 跑的，以后的生成对比以 DeepSeek 为准。

- **项目目标调整（2026-11-06）**：
  - 目标从"意图路由的固定流水线 GraphRAG"改为"自主规划检索路径的问答 Agent"：混合检索、图谱查询、章节原文读取、章节摘要封装为工具，LangGraph 实现"规划 → 检索 → 反思 → 作答"循环，以证据笔记和章节摘要作为分层记忆；与固定流程 RAG、闭卷基线对比。
  - 调整：章节摘要提前到 P3；引用校验提前到 P4；评测集 v1 与 P3 并行；Agent 的检索指标用"同等证据预算下的最终证据召回 + 过程覆盖率 + 引用准确率"，MRR 只用于固定 RAG。
  - **最终语料用全五卷**：全五卷导入和索引放在 P2；图谱先做卷一，再扩展到全五卷。**评测集 v1 覆盖全五卷（2026-11-06 用户确认）**，含跨卷题；P4 起的主设置是"全五卷 + 每题自己的阅读进度过滤"。
  - 出题仍由 Claude 按规范出（被测模型是 DeepSeek，两者分开）；judge 用 DeepSeek 时要用人工打分验证一致率；闭卷基线必做。
  - 暂停或降级：意图路由与问题分解的固定流程、Ragas、FastAPI / SSE 服务化（降级为可选演示）、插图提取与理解、pdf 与 md 解析器、社区发现、Text2Cypher。格式要求放宽为"解析器可插拔，支持 epub 和 txt"。
  - 简历定稿版本见本次会话：项目简介 + 混合检索与知识图谱 / Agent 规划与工具调用 / 上下文与记忆管理 / 评测体系四段。

- **P2 结论（2026-11-06）**：
  - **固定流程 RAG 基线**（config 默认值）：BM25（IK，`bm25_scheme="ik"`）与向量各取前 50 → RRF k=60 取前 40 → bge-reranker-v2-m3 重排前 40 → 前 8 进上下文（`rerank_top_k`），不做 small-to-big（`expand_window=0`），上下文预算 4000 token，chunk `0c94b7e4`，编码文本不加标题前缀。Agent 与固定 RAG 共用这条链路（P4 的工具层直接复用）。
  - 消融表 v1 的主要发现（dev 60 题，配对 bootstrap 区间不含 0 的才算显著）：
    - RRF 比纯向量 R@20 +0.081、cov@4000 +0.108（显著）；BM25 单路与向量单路相当，但两者互补。
    - rerank 是最大收益：R@5 +0.128、MRR +0.118（显著）。候选 20 / 40 / 60 之间无显著差异，取 40（cov@4000 最高，延迟可接受）。
    - BM25 加 CJK bigram、RRF k=10/100、向量编码加标题前缀：都无显著差异，选简单的默认值。
    - **small-to-big ±1 在固定 token 预算下显著有害**：cov@4000 −0.228（邻块挤占预算，挤掉了排名靠后但含证据的块）。固定流程默认不扩展；"按需读上下文"留给 P4 Agent 的工具（读邻近块 / 章节原文），由模型决定何时扩展。
    - 分块 300 / 800 字与默认 500 字比 cov@4000 无显著差异（800 字 −0.081 接近显著），保留 500 字。
    - **注释是否入索引的消融：不做**。注释从正文剥离单独存放，在 `text.txt` 中没有位置，按字符区间判定的评测永远不会把注释块算作命中，消融测不出收益；全五卷共 26 条注释，体量也太小。留到 P4 作为 Agent 可调用的工具再评估。
  - **设置 B（全五卷不过滤）**：比设置 A 显著下降（R@5 −0.072、R@20 −0.131、cov@4000 −0.156）。抽查排在金标前的 20 条范围外命中（`scripts/spotcheck_scope.py`，报告在 `data/reports/`）：能回答 1、可能 1、不能 18，假漏检约 5%–10%，下降主要是真实的跨卷噪声。20 条中 6 条是其他卷的附录名表（人名密集，容易被人物类问题召回）。P3 / P4 要考虑：Agent 按需过滤 `section_type=appendix`，或把附录当作专门的人物表工具。
  - **BM25 可复现**：ES 同分结果的顺序不确定，且覆盖写入留下的已删除文档仍计入文档频率，重建索引后分数会变（P2 早期同一配置 MRR 0.438 / 0.435）。现在搜索同分按 `chunk_id` 排序，每次同步后合并成单段；同一配置两次运行命中列表逐条一致。消融表 v1 全部基于修复后的 commit `59bf7ab` 重跑。
  - rerank 延迟：独占 GPU 时约 16 ms / 候选（40 候选 P50 约 630 ms）；batch 大小、max_length 对速度无明显影响。GPU 被其他进程占用时显存溢出到共享内存，速度会下降上百倍（P2 期间遇到过一次），跑评测时要独占 GPU。
  - txt 解析器（`txt-1`）：编码依次尝试 BOM / UTF-8 / GB18030 / Big5；自动判断一行一段、空行分段、硬换行三种分段方式；卷章标题用通用规则（第X卷 / 章、关键词）+ 启发式（无编号人名标题：一行一段时看前面的空行，其他分段方式看重复次数）。不含书籍专属知识。全五卷 epub 导出 txt 再导入，正文逐字一致、章节表一致。
  - 新增评测能力：cov@B（固定 token 预算下的证据覆盖率，B = 2000 / 4000 / 8000）、按题 bootstrap 置信区间、配对差值、分阶段延迟、消融表生成（`scripts/ablation_table.py --spec`）。

## 待决事项（P0 内需要确定）

- [x] 创建 `.wslconfig`（见决策记录）
- [x] Kimi 支持的结构化输出方式：`response_format={"type":"json_schema"}` 可用，kimi-k2.6 首次请求即通过 pydantic 校验；`json_object` 作为备选保留在 config
- [x] Kimi 账户的限流额度：实测 429 报错显示组织级上限为 **RPM 3**（最低等级）。`llm/` 新增按 RPM 节流的滑动窗口限流（`LLM__MAX_RPM`，默认不限）；使用 Kimi 时在 `.env` 中设 `LLM__MAX_RPM=3`。P3 全量抽取前需要充值升级等级，否则数千次调用不可行
