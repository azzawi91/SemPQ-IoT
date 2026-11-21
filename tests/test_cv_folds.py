import numpy as np

from src.data import mitbih
from src.l1_semantic.data import SplitArrays, concat_splits, subset_records


def test_group_folds_partition_and_determinism():
    rng = np.random.default_rng(0)
    counts = {r: rng.integers(0, 500, 5) for r in mitbih.DS1}
    f1 = mitbih.group_folds(counts, 4)
    f2 = mitbih.group_folds(dict(reversed(list(counts.items()))), 4)
    assert f1 == f2
    flat = [r for f in f1 for r in f]
    assert sorted(flat) == sorted(mitbih.DS1) and len(flat) == len(set(flat))


def test_subset_records_is_patient_disjoint():
    d = SplitArrays(np.zeros((6, 1, 4), np.float32), np.zeros((6, 5), np.float32),
                    np.arange(6), np.array([1, 1, 2, 2, 3, 3]))
    both = concat_splits([d, d])
    a, b = subset_records(both, [2], keep=False), subset_records(both, [2], keep=True)
    assert set(a.record) == {1, 3} and set(b.record) == {2} and len(a.y) + len(b.y) == 12
