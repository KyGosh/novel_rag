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
    """检索漏斗参数（P2 起使用，先占位）。"""

    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    fusion_top_k: int = 40
    rerank_top_k: int = 8


class PathsConfig(_Frozen):
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    results_dir: Path = PROJECT_ROOT / "eval" / "results"
    models_dir: Path = PROJECT_ROOT / "models"  # 本地模型权重（gitignore）

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

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
    paths: PathsConfig = PathsConfig()

    def snapshot(self) -> dict[str, Any]:
        """去掉密钥的配置快照，随每次实验结果一起保存。SecretStr 字段输出为 '**********'。"""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
