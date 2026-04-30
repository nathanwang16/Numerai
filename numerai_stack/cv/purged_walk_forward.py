"""Purged walk-forward CV aligned with Numerai's own scheme.

Numerai benchmark models use:
- 156-era prediction chunks
- embargo = 8 eras for 20D targets, 16 eras for 60D targets
- train on [first_era, first_chunk_era - embargo]
  predict on [first_chunk_era, first_chunk_era + 155]

Ref: https://docs.numer.ai/numerai-tournament/models
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_eras: tuple[str, ...]
    test_eras: tuple[str, ...]
    embargo_eras: tuple[str, ...]


def _to_sorted_list(eras: Iterable) -> list[str]:
    return sorted({str(e) for e in eras})


def purged_walk_forward_splits(
    eras: Sequence,
    chunk_size: int = 156,
    embargo: int = 8,
    min_train_eras: int = 52,
) -> list[WalkForwardSplit]:
    """Generate Numerai-style walk-forward splits.

    Parameters
    ----------
    eras : sequence of era labels (e.g. strings or ints); sort order = chronological.
    chunk_size : 156 (= 3 years of weekly eras).
    embargo : 8 for 20D targets, 16 for 60D targets.
    min_train_eras : skip chunks where training window would be shorter than this.
    """
    era_list = _to_sorted_list(eras)
    n = len(era_list)
    splits: list[WalkForwardSplit] = []
    fold = 0
    start = 0
    while start + chunk_size <= n or (start < n and fold == 0 and chunk_size > n):
        end = min(start + chunk_size, n)
        first_chunk_idx = start
        purge_start = max(0, first_chunk_idx - embargo)
        train_idx_end = purge_start  # exclusive
        if train_idx_end < min_train_eras:
            start = end
            fold += 1
            continue
        train = tuple(era_list[:train_idx_end])
        embargo_eras = tuple(era_list[train_idx_end:first_chunk_idx])
        test = tuple(era_list[first_chunk_idx:end])
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_eras=train,
                test_eras=test,
                embargo_eras=embargo_eras,
            )
        )
        fold += 1
        start = end
    # Special case: dataset smaller than chunk_size (e.g. validation-only).
    if not splits and n > min_train_eras + embargo:
        split = WalkForwardSplit(
            fold=0,
            train_eras=tuple(era_list[: n - embargo - chunk_size if chunk_size < n else min_train_eras]),
            test_eras=tuple(era_list[-chunk_size:]) if chunk_size <= n else tuple(),
            embargo_eras=tuple(),
        )
        splits.append(split)
    return splits


def combinatorial_purged_splits(
    eras: Sequence,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 8,
) -> Iterator[WalkForwardSplit]:
    """Combinatorial purged K-fold CV (Lopez de Prado).

    Splits the era axis into ``n_groups`` contiguous buckets. Every combination
    of ``n_test_groups`` buckets becomes a test set; the remaining buckets form
    the training set with a ``embargo``-era purge around each test bucket.
    Useful for hyperparameter search on overlapping 20D/60D labels.
    """
    era_list = _to_sorted_list(eras)
    n = len(era_list)
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("Require 1 <= n_test_groups < n_groups")
    bucket_size = n // n_groups
    buckets = [era_list[i * bucket_size : (i + 1) * bucket_size] for i in range(n_groups)]
    if n % n_groups:
        buckets[-1].extend(era_list[n_groups * bucket_size :])

    bucket_index = {era: gi for gi, b in enumerate(buckets) for era in b}
    fold = 0
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_eras: list[str] = []
        for gi in test_ids:
            test_eras.extend(buckets[gi])
        test_set = set(test_eras)

        # Purge: drop any training era within ``embargo`` positions of a test era.
        purge_set: set[str] = set()
        for gi in test_ids:
            bucket_eras = buckets[gi]
            first_idx = era_list.index(bucket_eras[0])
            last_idx = era_list.index(bucket_eras[-1])
            lo = max(0, first_idx - embargo)
            hi = min(n, last_idx + embargo + 1)
            for idx in range(lo, hi):
                if era_list[idx] not in test_set:
                    purge_set.add(era_list[idx])

        train_eras = tuple(e for e in era_list if e not in test_set and e not in purge_set)
        yield WalkForwardSplit(
            fold=fold,
            train_eras=train_eras,
            test_eras=tuple(test_eras),
            embargo_eras=tuple(sorted(purge_set)),
        )
        fold += 1


__all__ = [
    "WalkForwardSplit",
    "purged_walk_forward_splits",
    "combinatorial_purged_splits",
]
