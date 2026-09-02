import numpy as np


class FDRUtils:
    """Target-decoy FDR utilities (label: target=1, decoy=0)."""

    @staticmethod
    def calculate_q_values(score, label):
        scores = np.asarray(score, dtype=float)
        labels = np.asarray(label, dtype=int)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
            raise ValueError("score and label must be one-dimensional and have equal length")
        if len(scores) == 0:
            return np.asarray([], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("score contains NaN or infinite values")

        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        targets = np.cumsum(sorted_labels == 1)
        decoys = np.cumsum(sorted_labels == 0)
        fdr = decoys / np.maximum(targets, 1)

        # Equal scores receive the same (worst cumulative) FDR before the
        # reverse cumulative minimum is applied.
        boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
        starts = np.r_[0, boundaries[:-1] + 1]
        tied_fdr = fdr.copy()
        for start, end in zip(starts, boundaries):
            tied_fdr[start:end + 1] = fdr[end]
        q_sorted = np.minimum.accumulate(tied_fdr[::-1])[::-1]
        q_values = np.empty_like(q_sorted, dtype=float)
        q_values[order] = q_sorted
        return q_values

    def calculate_fdr_list(self, score, label):
        return self.calculate_q_values(score, label).tolist()

    def calculate_fdr(self, score, label, target_fdr=0.01, top_n=0):
        if top_n:
            raise ValueError("top_n decoy removal is no longer supported")
        q_values = self.calculate_q_values(score, label)
        labels = np.asarray(label, dtype=int)
        scores = np.asarray(score, dtype=float)
        passed = (q_values <= float(target_fdr)) & (labels == 1)
        threshold = float(scores[passed].min()) if passed.any() else float("inf")
        return int(passed.sum()), threshold
