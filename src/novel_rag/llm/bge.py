"""本地 GPU 上的 bge-m3（Embedding）和 bge-reranker-v2-m3（Reranker）。整个项目只有本文件 import FlagEmbedding。

模型在进程内只加载一次（由 factory 保证单例），常驻显存。编码按 batch_size 分批，长文本批量编码时
要按 8GB 显存调小 batch_size / max_length。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from novel_rag.config import EmbeddingConfig, RerankerConfig
from novel_rag.llm.base import Embedder, Reranker, l2_normalize


def _module_device(obj: object) -> str:
    """找出 FlagEmbedding 包装对象里 torch 模型实际所在的设备（用于冒烟检查）。"""
    import torch

    for name in ("model", "_model"):
        m = getattr(obj, name, None)
        if isinstance(m, torch.nn.Module):
            return str(next(m.parameters()).device)
    return "unknown"


class BgeM3Embedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig, load_from: str | None = None):
        """load_from：实际加载路径（本地目录或 Hub 名），默认等于 cfg.model_name。"""
        from FlagEmbedding import BGEM3FlagModel

        self.cfg = cfg
        self.model_name = cfg.model_name
        self.dim = cfg.dim
        self._model = BGEM3FlagModel(load_from or cfg.model_name, use_fp16=cfg.fp16, devices=cfg.device)

    @property
    def device(self) -> str:
        return _module_device(self._model)

    def _encode(self, texts: Sequence[str], *, query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        fn = self._model.encode_queries if query else self._model.encode_corpus
        out = fn(
            list(texts),
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = l2_normalize(np.asarray(out["dense_vecs"]))
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if vecs.shape[1] != self.dim:
            raise ValueError(f"{self.model_name} 输出维度 {vecs.shape[1]}，与配置 dim={self.dim} 不一致")
        return vecs

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], query=True)[0]


class BgeReranker(Reranker):
    def __init__(self, cfg: RerankerConfig, load_from: str | None = None):
        from FlagEmbedding import FlagReranker

        self.cfg = cfg
        self.model_name = cfg.model_name
        self._model = FlagReranker(load_from or cfg.model_name, use_fp16=cfg.fp16, devices=cfg.device)

    @property
    def device(self) -> str:
        return _module_device(self._model)

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        scores = self._model.compute_score(
            [(query, c) for c in candidates],
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            normalize=True,
        )
        if isinstance(scores, float | int):
            scores = [scores]
        return [float(s) for s in scores]


def gpu_info() -> dict[str, object]:
    """torch 看到的 GPU 信息（环境检查用）。"""
    import torch

    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            name=props.name,
            total_mem_gb=round(total / 1024**3, 2),
            free_mem_gb=round(free / 1024**3, 2),
            allocated_by_torch_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        )
    return info
