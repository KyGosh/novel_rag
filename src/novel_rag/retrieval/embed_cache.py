"""文档向量缓存：键为"模型名 + 编码文本的 sha256"，换 collection（新的 chunk 版本、新索引参数）时无需重新编码。"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from novel_rag.llm.base import Embedder


def encode_text(chunk: dict[str, Any], template: str, book_title: str = "") -> str:
    """送入 embedding 模型的文本。title_prefix 只影响向量，不影响 chunk 的字符偏移。"""
    if template == "raw":
        return chunk["text"]
    if template == "title_prefix":
        prefix = " ".join(p for p in (book_title, chunk.get("chapter_title") or "") if p)
        return f"{prefix}\n{chunk['text']}" if prefix else chunk["text"]
    raise ValueError(f"未知的编码模板：{template}")


class EmbeddingCache:
    def __init__(self, embedder: Embedder, store: MutableMapping[str, Any]):
        self.embedder = embedder
        self.store = store
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return f"{self.embedder.model_name}:{self.embedder.dim}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        out = np.zeros((len(texts), self.embedder.dim), dtype=np.float32)
        todo = []
        for i, k in enumerate(keys):
            cached = self.store.get(k)
            if cached is None:
                todo.append(i)
            else:
                out[i] = np.frombuffer(cached, dtype=np.float32)
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            vecs = self.embedder.embed_documents([texts[i] for i in todo])
            for i, v in zip(todo, vecs, strict=True):
                out[i] = v
                self.store[keys[i]] = np.asarray(v, dtype=np.float32).tobytes()
        return out


def open_embedding_store(directory: Path) -> MutableMapping[str, Any]:
    import diskcache

    directory.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(directory), size_limit=4 * 1024**3)
