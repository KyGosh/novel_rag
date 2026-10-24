# 进度与决策记录

新会话开始时先读本文件、`.claude/CLAUDE.md` 和当前阶段的 `docs/phases/Px.md`；需要模块细节时查 `docs/design-notes.md`。每个阶段结束时更新"当前状态"和"决策记录"。

## 当前状态

- 阶段：P0（骨架）基本完成，方案见 `docs/phases/P0.md`
  - 已完成：uv 项目骨架（src 布局，各模块空包）、`docker-compose.yml` + ES+IK 镜像、`config.py`（含去密钥快照）、`llm/` 抽象层（接口 / Fake / 缓存 / 重试限流 / 用量 / 结构化输出 / bge 本地模型 / Kimi）、单测 26 项全过、`scripts/check_env.py` 7 项全部通过
  - 剩余：原文放入 `data/raw/`（P1 开始前完成）；P3 前在控制台确认 Kimi 账户限流等级
  - 下一阶段：P1（多格式解析、分块、Milvus 向量 baseline、评测集 v0）
- 最新评测数字：暂无（P1 产出 baseline）

## 开发阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | docker-compose（ES+IK / Milvus / Neo4j）、uv 项目骨架、config、`llm/` 抽象层、冒烟测试 | 进行中 |
| P1 | 多格式解析（txt / md / epub / 文字层 pdf）+ 插图提取与过滤 + 分块（卷一，不含章节摘要）+ Milvus 向量 baseline + 最简生成 + 评测集 v0 + Recall/MRR | 未开始 |
| P2 | ES BM25 + RRF + Rerank，消融表 v1 | 未开始 |
| P3 | 知识图谱构建（抽取、消歧、Neo4j、回灌 IK 词典） | 未开始 |
| P4 | 实体链接 + 图检索 + LangGraph 编排 + 章节摘要 | 未开始 |
| P5 | 引用、拒答、防剧透，Ragas，全量消融；扩展到全五卷；插图理解（视觉模型生成描述并入库，可选增强） | 未开始 |
| P6 | FastAPI、SSE、异步、缓存、tracing、性能优化 | 未开始 |
| P7 | README、消融表、bad case 分析、演示 | 未开始 |

## 决策记录

- 按阶段 P0→P7 推进，模块分阶段逐步实现，不按模块顺序开发。
- 每个阶段开一个新会话，靠本文件交接；subagent 只用于查资料和代码审查。
- 计划和方案阶段只讲思路，不写详细代码。
- 目标书籍为《冰与火之歌》中文版全五卷，P1–P3 先只用卷一。原文受版权保护，只在本地处理，不入库。
- LLM 选用 Kimi（Moonshot，OpenAI 兼容接口）。
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

## 待决事项（P0 内需要确定）

- [x] 创建 `.wslconfig`（见决策记录）
- [x] Kimi 支持的结构化输出方式：`response_format={"type":"json_schema"}` 可用，kimi-k2.6 首次请求即通过 pydantic 校验；`json_object` 作为备选保留在 config
- [ ] Kimi 账户的限流额度：API 响应头里没有限流信息，需要在开放平台控制台查看账户等级（当前只有 15 元代金券、现金余额 0，大概率是最低等级）。P3 全量抽取前按等级设置 `LLM__MAX_CONCURRENCY`（当前默认 4）
