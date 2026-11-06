"""全局配置：所有参数（地址、模型名、top_k、阈值）集中在这里，从 `.env` / 环境变量读取。

嵌套键用双下划线分隔，例如 `LLM__API_KEY`、`LLM__TASKS__GENERATION__MODEL`。
业务代码通过 `get_settings()` 取得进程内唯一的只读实例。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ESConfig(_Frozen):
    url: str = "http://localhost:9200"
    request_timeout: float = 30.0


class MilvusConfig(_Frozen):
    uri: str = "http://localhost:19530"
    token: SecretStr = SecretStr("")
    db_name: str = "default"


class Neo4jConfig(_Frozen):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")
    database: str = "neo4j"


class LLMTaskConfig(_Frozen):
    """某一类任务（抽取、生成、judge……）使用的模型及请求参数。"""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    # 厂商专属参数原样透传，例如 Kimi 的 {"thinking": {"type": "disabled"}}
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _default_tasks() -> dict[str, LLMTaskConfig]:
    no_thinking = {"thinking": {"type": "disabled"}}
    return {
        "extraction": LLMTaskConfig(model="kimi-k2.6", extra_body=no_thinking),
        "generation": LLMTaskConfig(model="kimi-k2.6", extra_body=no_thinking),
        "judge": LLMTaskConfig(model="kimi-k2.6", extra_body=no_thinking),
        "smoke": LLMTaskConfig(model="kimi-k2.6", max_tokens=256, extra_body=no_thinking),
    }


class LLMConfig(_Frozen):
    base_url: str = "https://api.moonshot.cn/v1"
    api_key: SecretStr = SecretStr("")
    max_concurrency: int = 4
    max_rpm: int | None = None  # 账户每分钟请求数上限（Kimi 最低等级为 3）；None 表示不节流，429 交给重试
    timeout: float = 120.0
    max_retries: int = 5  # 429 / 5xx / 网络错误的指数退避重试次数
    # 结构化输出：json_schema 由服务端按 schema 约束；json_object 只保证合法 JSON，schema 写进 prompt
    structured_mode: Literal["json_schema", "json_object"] = "json_schema"
    structured_max_attempts: int = 3  # pydantic 校验失败后的重新请求次数（含首次）
    cache_enabled: bool = True
    tasks: dict[str, LLMTaskConfig] = Field(default_factory=_default_tasks)

    @field_validator("tasks", mode="before")
    @classmethod
    def _merge_default_tasks(cls, v: Any) -> Any:
        """环境变量只覆盖某个任务的某个字段时，其余任务和字段保留默认值。"""
        if not isinstance(v, dict):
            return v
        merged: dict[str, Any] = {k: t.model_dump() for k, t in _default_tasks().items()}
        for name, override in v.items():
            if isinstance(override, LLMTaskConfig):
                override = override.model_dump()
            merged[name] = {**merged.get(name, {}), **override}
        return merged


class EmbeddingConfig(_Frozen):
    model_name: str = "BAAI/bge-m3"
    dim: int = 1024
    batch_size: int = 16
    max_length: int = 1024
    device: str = "cuda"
    fp16: bool = True


class RerankerConfig(_Frozen):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 1024
    batch_size: int = 16
    device: str = "cuda"
    fp16: bool = True


class RetrievalConfig(_Frozen):
    """检索漏斗参数（BM25 / RRF / rerank 在 P2 起使用）。"""

    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    fusion_top_k: int = 40
    rerank_top_k: int = 8
    # Milvus HNSW 索引与搜索参数
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 128
    metric_type: str = "COSINE"
    # 编码文本：raw = chunk 原文；title_prefix = 前面加"卷名 章节标题"（只影响向量，不影响字符偏移）
    embed_text_template: Literal["raw", "title_prefix"] = "raw"
    milvus_upsert_batch: int = 256


class ChunkConfig(_Frozen):
    """切句与分块参数。参数（连同切句规则版本）的 hash 就是 chunk 版本号，任何改动都会产生新版本。"""

    target_chars: int = 500
    max_chars: int = 800
    min_chars: int = 150  # 章末短于此的尾块并入前一块
    overlap_chars: int = 90  # 重叠只取整句，总长不超过此值
    long_quote_chars: int = 200  # 引号内的对话超过此长度，允许在句末标点处断开
    sentence_hard_max: int = 300  # 超长单句在逗号 / 分号处硬切
    query_max_tokens: int = 64  # token 预算检查：P99(chunk token) + 查询长度 ≤ reranker.max_length


class EvalConfig(_Frozen):
    """检索评测参数。"""

    # 证据区间与 chunk 的重叠字数 ≥ min(区间长度 × ratio, chars) 判定命中
    relevance_min_ratio: float = 0.5
    relevance_min_chars: int = 50
    ks: tuple[int, ...] = (5, 10, 20)
    generate_top_k: int = 6  # 最简生成取前几个 chunk


class SectionKeywords(_Frozen):
    """section_type 判定用的通用关键词（按标题前缀匹配，不区分大小写）。不是书籍专属知识，书籍配置可以覆盖判定结果。"""

    prologue: tuple[str, ...] = ("序幕", "序章", "楔子", "引子", "prologue")
    epilogue: tuple[str, ...] = ("终章", "尾声", "epilogue")
    appendix: tuple[str, ...] = ("附录", "appendix")
    front_matter: tuple[str, ...] = (
        "出版说明", "作者简介", "作者介绍", "关于作者", "内容简介", "译者序", "译序", "前言", "序言", "目录", "版权",
        "preface", "foreword", "contents", "copyright",
    )  # fmt: skip
    back_matter: tuple[str, ...] = ("后记", "译后记", "致谢", "鸣谢", "afterword", "acknowledg")
    dedication: tuple[str, ...] = ("献给", "谨以此书", "献辞", "dedicated to", "dedication")


class IngestConfig(_Frozen):
    # 不可信输入的解压限制
    max_archive_files: int = 5000
    max_uncompressed_bytes: int = 512 * 1024 * 1024
    max_member_bytes: int = 64 * 1024 * 1024
    # 章节标题是人名（POV）的启发式：标题不超过 N 字，且在全系列中作为章节标题出现至少 M 次
    pov_title_max_len: int = 6
    pov_min_occurrences: int = 3
    # 去掉标题后正文少于该字数的章节视为空白页（只有标题或插图）
    blank_section_max_chars: int = 10
    # 写入 text.txt 的 section_type；其余（标题页、献词、前后辅文）只出现在质检报告中
    text_section_types: tuple[str, ...] = ("narrative", "appendix")
    # 卷启用范围（P1 只索引卷一）；None 表示全部
    enabled_volumes: tuple[int, ...] | None = (1,)
    # 分册中不夹在两个汉字之间的半角 '?' 占比达到该值，就认为它用作问号（统一为全角），而不是丢失的间隔号
    qmark_punct_ratio: float = 0.2
    keywords: SectionKeywords = SectionKeywords()
    qc_sample_size: int = 8
    qc_qmark_sample_size: int = 20


class EvalGenConfig(_Frozen):
    """评测集构建：出题窗口采样、组合题配对、证据定位、预筛与冻结的参数。"""

    seed: int = 20261025
    window_min_chars: int = 600
    window_max_chars: int = 900
    include_appendix: bool = False  # 附录（谱系表等）默认不参与生成式出题，留给人工题
    # 第二轮组合题的任务数：人物关系 / 多跳 / 跨章节情节
    combo_counts: dict[str, int] = Field(default_factory=lambda: {"relation": 60, "multihop": 70, "plot": 60})
    plot_min_chapter_gap: int = 5  # 跨章节题相邻两个窗口至少相隔几章
    plot_windows: int = 3  # 跨章节题最多用几个窗口
    max_window_uses: int = 3  # 同一题型中一个窗口最多参与几个组合任务
    # 证据定位
    quote_min_chars: int = 6
    quote_max_chars: int = 200
    fuzzy_threshold: float = 90.0  # rapidfuzz 局部对齐分数（0–100）
    # 预筛
    copy_lcs_chars: int = 8  # 问题与证据的最长公共子串达到该长度，标记为照抄
    dup_jaccard: float = 0.8  # 两道题字符 bigram 的 Jaccard 相似度达到该值，标记为近似重复
    answer_max_chars: int = 40
    question_max_chars: int = 80
    # 冻结
    per_type_target: int = 30
    copyright_max_substring: int = 20  # 问题和答案中不得出现原文中这么长的连续片段


class PathsConfig(_Frozen):
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    results_dir: Path = PROJECT_ROOT / "eval" / "results"
    models_dir: Path = PROJECT_ROOT / "models"  # 本地模型权重（gitignore）

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.cache_dir / "embeddings"

    @property
    def eval_work_dir(self) -> Path:
        """评测集构建的中间产物（出题任务、候选题含原文片段、校验日志），属于 data/，不入库。"""
        return self.data_dir / "eval_work"

    @property
    def datasets_dir(self) -> Path:
        return self.results_dir.parent / "datasets"

    def resolve_model(self, model_name: str) -> str:
        """模型加载位置：model_name 本身是存在的路径 → 直接用；models_dir 下有同名目录
        （取 "BAAI/bge-m3" 的最后一段）→ 用本地目录；否则原样返回，由 HuggingFace Hub 下载。"""
        if Path(model_name).is_dir():
            return model_name
        local = self.models_dir / model_name.rsplit("/", 1)[-1]
        return str(local) if local.is_dir() else model_name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",  # .env 里还有 compose 专用的变量
        frozen=True,
    )

    es: ESConfig = ESConfig()
    milvus: MilvusConfig = MilvusConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chunk: ChunkConfig = ChunkConfig()
    eval: EvalConfig = EvalConfig()
    ingest: IngestConfig = IngestConfig()
    evalgen: EvalGenConfig = EvalGenConfig()
    paths: PathsConfig = PathsConfig()

    def snapshot(self) -> dict[str, Any]:
        """去掉密钥的配置快照，随每次实验结果一起保存。SecretStr 字段输出为 '**********'。"""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
