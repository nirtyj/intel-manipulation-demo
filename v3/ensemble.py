"""ACT's temporal ensembling as a chunk smoother for physicalai's runtime action queue.

The policy is queried every tick (``SyncExecution(request_threshold=1.0)``); every future step's action is
the weighted mean of all the chunks that have predicted it so far, the k-th prediction for a step weighted
``exp(-coeff * k)`` (Zhao et al., ACT, 2023). The queue pops from the front between merges, so the running
weights of the remaining actions are the tail of the previous merge's.
"""

from __future__ import annotations

import numpy as np
from physicalai.runtime.smoothers import ChunkSmoother


class TemporalEnsembleSmoother(ChunkSmoother):
    def __init__(self, coeff: float = 0.01) -> None:
        self.coeff = coeff
        self._wsum = np.zeros(0)
        self._count = np.zeros(0)

    def merge(self, remaining: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        n_r = len(remaining)
        wsum, count = self._wsum[len(self._wsum) - n_r:], self._count[len(self._count) - n_r:]
        if len(wsum) != n_r:  # queue was cleared or reset: start over
            wsum, count = np.ones(n_r), np.ones(n_r)
        n = min(n_r, len(incoming))
        w = np.exp(-self.coeff * count[:n])[:, None]
        blended = (remaining[:n] * wsum[:n, None] + incoming[:n] * w) / (wsum[:n, None] + w)
        out = np.concatenate([blended, incoming[n:]], axis=0).astype(np.float32)
        self._wsum = np.concatenate([wsum[:n] + w[:, 0], np.ones(len(incoming) - n)])
        self._count = np.concatenate([count[:n] + 1, np.ones(len(incoming) - n)])
        return out
