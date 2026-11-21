import json

import pandas as pd
import pytest

from src.data import mitbih, ptbxl, splits


def _scp():
    return pd.DataFrame(
        {"diagnostic": [1, 1, 1, 0, 1], "diagnostic_class": ["NORM", "MI", "STTC", None, "HYP"]},
        index=["NORM", "IMI", "NDT", "SR", "LVH"],
    )


def _db(fold_of_patient):
    rows = [
        (1, 10, {"NORM": 100.0, "SR": 0.0}, "records100/00000/00001_lr"),
        (2, 10, {"IMI": 50.0, "NDT": 100.0}, "records100/00000/00002_lr"),
        (3, 11, {"SR": 0.0}, "records100/00000/00003_lr"),  # no superclass -> excluded
        (4, 12, {"LVH": 0.0}, "records100/00000/00004_lr"),
    ]
    df = pd.DataFrame(rows, columns=["ecg_id", "patient_id", "scp_codes", "filename_lr"]).set_index("ecg_id")
    df["strat_fold"] = df["patient_id"].map(fold_of_patient)
    return df


def test_superclass_labels_and_folds():
    folds = ptbxl.build_folds(_db({10: 1, 11: 9, 12: 10}), _scp())
    assert folds["ecg_id"].tolist() == [1, 2, 4]
    assert folds.set_index("ecg_id").loc[2, ["MI", "STTC", "NORM"]].tolist() == [1, 1, 0]
    assert folds["split"].tolist() == ["train", "train", "test"]


def test_patient_leakage_detected():
    db = _db({10: 1, 11: 9, 12: 10})
    db.loc[2, "strat_fold"] = 10  # same patient 10 now in train and test
    with pytest.raises(AssertionError, match="leakage"):
        ptbxl.build_folds(db, _scp())


def test_split_files_hash_roundtrip(tmp_path):
    folds = ptbxl.build_folds(_db({10: 1, 11: 9, 12: 10}), _scp())
    sd = splits.mitbih_split_dict((101, 106, 108), {"method": "test"})
    sums = splits.write_splits(sd, folds, tmp_path)
    assert splits.verify_splits(tmp_path) == sums
    (tmp_path / splits.PTBXL_FILE).write_text("tampered\n")
    with pytest.raises(RuntimeError, match="changed"):
        splits.verify_splits(tmp_path)


def test_validation_records_must_come_from_ds1():
    with pytest.raises(ValueError):
        splits.mitbih_split_dict((100,), {})


@pytest.mark.skipif(not (splits.SPLIT_DIR / splits.SUMS_FILE).exists(), reason="split files not built yet")
def test_committed_split_files():
    splits.verify_splits()
    s = json.loads((splits.SPLIT_DIR / splits.MITBIH_FILE).read_text())
    assert set(s["train"]) | set(s["val"]) == set(mitbih.DS1)
    assert not set(s["train"]) & set(s["val"])
    assert s["test"] == list(mitbih.DS2)
    folds = splits.load_ptbxl_folds()
    ptbxl.assert_patient_disjoint(folds)
    assert set(folds.loc[folds.split == "test", "strat_fold"]) == {10}
    assert set(folds.loc[folds.split == "val", "strat_fold"]) == {9}
    assert folds["ecg_id"].is_unique


def test_physionet_sums_verification_with_exclusion(tmp_path):
    import hashlib
    from src.data import physionet

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "records500").mkdir()
    h = hashlib.sha256(b"x").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{h} a.txt\n{'0' * 64} records500/b.dat\n")
    s = physionet.verify_against_sums(tmp_path, ("records500/",))
    assert (s["files_checked"], s["files_excluded"], s["missing"], s["mismatched"]) == (1, 1, [], [])
    s = physionet.verify_against_sums(tmp_path)
    assert s["missing"] == ["records500/b.dat"]
    (tmp_path / "a.txt").write_text("y")
    assert physionet.verify_against_sums(tmp_path, ("records500/",))["mismatched"] == ["a.txt"]
