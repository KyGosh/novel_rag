"""置信区间：按题目做百分位 bootstrap。

评测集每类只有十几道题，按题型的差异很容易是噪声，所以每个均值都附置信区间；比较两个配置时用配对 bootstrap
（同一批题、同一组重采样下标上的差值），比"两个区间是否重叠"更灵敏，也更符合"每行只改一处"的消融设计。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _resample_means(values: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_samples, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(values: Sequence[float], *, n_samples: int, seed: int, level: float) -> tuple[float, float]:
    """均值的百分位 bootstrap 置信区间；样本为空时返回 (nan, nan)。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    means = _resample_means(arr, n_samples, seed)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_diff(a: Sequence[float], b: Sequence[float], *, n_samples: int, seed: int, level: float) -> dict[str, float]:
    """b − a 的均值差及其配对 bootstrap 置信区间；p_le0 是重采样中差值 ≤ 0 的比例（单侧，越小越说明 b 更好）。"""
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if da.shape != db.shape:
        raise ValueError("配对比较的两组题目必须一一对应")
    if da.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le0": float("nan")}
    d = db - da
    means = _resample_means(d, n_samples, seed)
    alpha = (1 - level) / 2
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1 - alpha)),
        "p_le0": float((means <= 0).mean()),
    }
