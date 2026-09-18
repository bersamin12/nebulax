"""Splits: disjoint, covering where they should be, grouped where the protocol says grouped,
and the named presets resolve to the dates/groups the plan fixes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import splits as SP


@dataclass
class FakeData:
    """The duck-typed slice of BenchData that :func:`make_split` needs."""

    t_end: np.ndarray
    unit: np.ndarray
    group: np.ndarray
    labels: pd.DataFrame


def _fake(n=120, start="2020-02-01", freq="1D", **cols) -> FakeData:
    t = pd.date_range(start, periods=n, freq=freq).to_numpy().astype("datetime64[ms]")
    unit = np.array([f"T{i % 4:02d}" for i in range(n)], dtype=object)
    frame = pd.DataFrame({"train_id": unit, "run_id": [f"r{i % 8}" for i in range(n)], **cols})
    return FakeData(t_end=t, unit=unit, group=unit.copy(), labels=frame)


def _assert_disjoint_and_in_range(sp: SP.Split, n: int):
    parts = [sp.train, sp.val, sp.test]
    allidx = np.concatenate(parts)
    assert allidx.size == np.unique(allidx).size, "parts overlap"
    assert (allidx >= 0).all() and (allidx < n).all()


# --------------------------------------------------------------------------------------


def test_temporal_split_cuts_are_half_open_and_cover():
    t = pd.date_range("2020-02-01", periods=100, freq="1D").to_numpy().astype("datetime64[ms]")
    sp = SP.temporal_split(t, "2020-03-01", "2020-03-11")
    _assert_disjoint_and_in_range(sp, 100)
    sp.check(require_cover=True)
    assert sp.train.size == 29  # 1-29 Feb 2020 (leap year)
    assert sp.val.size == 10
    assert sp.test.size == 61
    # the boundary row belongs to the later part
    assert t[sp.val[0]] == np.datetime64("2020-03-01", "ms")


def test_temporal_split_rejects_unordered_cuts():
    t = pd.date_range("2020-02-01", periods=10, freq="1D").to_numpy().astype("datetime64[ms]")
    with pytest.raises(ValueError, match="ordered"):
        SP.temporal_split(t, "2020-03-01", "2020-02-01")


def test_split_check_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        SP.Split(name="bad", train=np.array([0, 1, 2]), val=np.array([2]), test=np.array([3]), n_rows=4)


def test_group_kfold_is_disjoint_covering_and_group_pure():
    g = np.array([f"b{i % 10}" for i in range(200)], dtype=object)
    folds = SP.group_kfold(g, k=5, seed=1)
    assert len(folds) == 5
    seen_test = []
    for sp in folds:
        _assert_disjoint_and_in_range(sp, 200)
        sp.check(require_cover=True)
        test_groups = set(g[sp.test])
        assert not (test_groups & set(g[sp.train])), "a group straddles the train/test fence"
        assert not (test_groups & set(g[sp.val]))
        seen_test.append(test_groups)
    assert set().union(*seen_test) == set(g), "every group is tested exactly once across folds"


def test_stratified_group_kfold_keeps_classes_in_every_test_fold():
    g = np.array([f"b{i % 12}" for i in range(240)], dtype=object)
    y = np.array(["healthy" if int(x[1:]) < 6 else "outer_race" for x in g], dtype=object)
    folds = SP.group_kfold(g, k=3, stratify=y, seed=0)
    for sp in folds:
        sp.check(require_cover=True)
        assert set(y[sp.test]) == {"healthy", "outer_race"}


def test_leave_one_group_out_holds_out_each_group_once():
    g = np.array(["20kg", "40kg", "neg40kg"] * 30, dtype=object)
    folds = SP.leave_one_group_out(g, seed=0)
    assert len(folds) == 3
    assert sorted(f.meta["held_out_group"] for f in folds) == ["20kg", "40kg", "neg40kg"]
    for sp in folds:
        sp.check(require_cover=True)
        assert set(g[sp.test]) == {sp.meta["held_out_group"]}
        assert sp.meta["held_out_group"] not in set(g[sp.train])


def test_leave_one_unit_out_is_the_same_machinery_with_the_protocol_name():
    u = np.array([f"T{i % 5:02d}" for i in range(50)], dtype=object)
    folds = SP.leave_one_unit_out(u, seed=3)
    assert len(folds) == 5 and folds[0].name == "leave_one_unit_out"
    for sp in folds:
        assert set(u[sp.test]).isdisjoint(set(u[sp.train]) | set(u[sp.val]))


def test_split_to_frame_round_trips_indices():
    sp = SP.temporal_split(
        pd.date_range("2020-02-01", periods=30, freq="1D").to_numpy().astype("datetime64[ms]"),
        "2020-02-15",
        "2020-02-20",
    )
    df = sp.to_frame()
    assert set(df["part"]) == {"train", "val", "test"}
    assert sorted(df.loc[df["part"] == "test", "row"]) == sorted(sp.test.tolist())
    assert len(SP.splits_to_frame([sp, sp])) == 2 * len(df)


# --------------------------------------------------------------------------------------
# named presets
# --------------------------------------------------------------------------------------


def test_metropt_presets_cut_where_the_protocol_says():
    data = _fake(n=220, start="2020-02-01", freq="1D")
    (sp,) = SP.make_split("metropt_temporal", data)
    t = data.t_end
    assert t[sp.train].max() < np.datetime64("2020-04-01", "ms")
    assert t[sp.val].min() >= np.datetime64("2020-04-01", "ms")
    assert t[sp.val].max() < np.datetime64("2020-04-11", "ms")
    assert t[sp.test].min() >= np.datetime64("2020-04-11", "ms")

    (cont,) = SP.make_split("metropt_contaminated", data)
    assert t[cont.train].max() < np.datetime64("2020-06-01", "ms")
    assert t[cont.test].min() >= np.datetime64("2020-06-04", "ms")


def test_cranfield_rep_preset_uses_rank_not_raw_id():
    # the release numbers reps 1..10, the protocol talks about ranks 0-6 / 7 / 8-9
    reps = np.repeat(np.arange(1, 11), 6)
    data = _fake(n=60, meta_rep=reps)
    (sp,) = SP.make_split("cranfield_rep", data)
    sp.check(require_cover=True)
    assert set(reps[sp.train]) == set(range(1, 8))
    assert set(reps[sp.val]) == {8}
    assert set(reps[sp.test]) == {9, 10}


def test_ottawa_preset_groups_by_bearing():
    bid = np.array([i % 10 for i in range(200)])
    ftype = np.where(bid < 5, "healthy", "outer_race")
    data = _fake(n=200, meta_bearing_id=bid, fault_type=ftype)
    folds = SP.make_split("ottawa_sgkf5", data)
    assert len(folds) == 5
    for sp in folds:
        sp.check(require_cover=True)
        assert set(bid[sp.test]).isdisjoint(set(bid[sp.train]))


def test_unknown_preset_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown split preset"):
        SP.make_split("nope", _fake())


def test_temporal_fracs_fallback_is_chronological():
    data = _fake(n=100)
    (sp,) = SP.make_split("temporal_fracs", data)
    sp.check(require_cover=True)
    assert data.t_end[sp.train].max() <= data.t_end[sp.val].min()
    assert data.t_end[sp.val].max() <= data.t_end[sp.test].min()


# --------------------------------------------------------------------------------------
# a preset may only be used on a dataset it is declared for
# --------------------------------------------------------------------------------------


@dataclass
class FakeNamedData(FakeData):
    """FakeData that also claims a real dataset name, so the preset gate applies."""

    dataset: str = "ottawa"


def _named(dataset: str, **kw) -> FakeNamedData:
    f = _fake(**kw)
    return FakeNamedData(t_end=f.t_end, unit=f.unit, group=f.group, labels=f.labels, dataset=dataset)


def test_temporal_fracs_is_refused_on_the_grouped_datasets():
    """`dataset: ottawa, split: temporal_fracs` used to be an accepted config: it puts the
    same bearing on both sides of the fence, and Ottawa's CLS target is constant per bearing,
    so that is direct label leakage. Same for Cranfield's reps."""
    for dataset in ("ottawa", "cranfield"):
        with pytest.raises(ValueError, match="not defined for dataset"):
            SP.make_split("temporal_fracs", _named(dataset, n=60))


def test_preset_gate_names_the_presets_that_do_apply():
    with pytest.raises(ValueError) as exc:
        SP.make_split("metropt_temporal", _named("ottawa", n=60))
    assert "ottawa_sgkf5" in str(exc.value)


def test_preset_gate_allows_the_declared_dataset_and_ignores_unknown_ones():
    (sp,) = SP.make_split("temporal_fracs", _named("metropt3", n=60))
    sp.check(require_cover=True)
    SP.check_preset_dataset("temporal_fracs", "fake_test_fixture")  # not policed
    SP.check_preset_dataset("temporal_fracs", None)


def test_every_preset_declares_at_least_one_known_dataset():
    for name, preset in SP.PRESETS.items():
        assert preset.datasets, name
        assert set(preset.datasets) <= SP.KNOWN_DATASETS, name


# --------------------------------------------------------------------------------------
# an AD split must never hand the runner an empty validation slice
# --------------------------------------------------------------------------------------


def test_leave_one_group_out_with_two_groups_still_carves_a_validation_slice():
    """Cranfield's leave-one-motion-profile-out has exactly two profiles, so the non-held-out
    side is a single group and whole-group carving is impossible. It used to return val = 0
    rows on both folds, and the runner then finished status='ok' with no threshold at all."""
    profile = np.array(["sin"] * 30 + ["trap"] * 30, dtype=object)
    data = _fake(n=60, meta_motion_profile=profile)
    folds = SP.make_split("cranfield_loo_profile", _named("cranfield", n=60, meta_motion_profile=profile))
    assert len(folds) == 2
    for sp in folds:
        sp.check(require_cover=True)
        assert sp.val.size > 0, "no validation rows -> no operating point"
        assert sp.train.size > 0
        assert sp.meta["val_carved_by"] == "row"
        # the held-out profile is still entirely out of both train and val
        assert set(profile[sp.test]).isdisjoint(set(profile[sp.train]))
        assert set(profile[sp.test]).isdisjoint(set(profile[sp.val]))
    assert data is not None


def test_group_kfold_carves_a_validation_slice_even_with_one_training_group():
    groups = np.array(["a"] * 20 + ["b"] * 20, dtype=object)
    folds = SP.group_kfold(groups, k=2, seed=0)
    for sp in folds:
        sp.check(require_cover=True)
        assert sp.val.size > 0
        assert sp.train.size > 0
        assert set(groups[sp.test]).isdisjoint(set(groups[sp.train]))


def test_no_preset_fold_leaves_an_empty_validation_slice():
    reps = np.repeat(np.arange(1, 11), 6)
    profile = np.array(["sin", "trap"] * 30, dtype=object)
    loads = np.array([20.0, 40.0, -40.0] * 20)
    data = _named("cranfield", n=60, meta_rep=reps, meta_motion_profile=profile, load_kg=loads)
    for preset in ("cranfield_rep", "cranfield_loo_load", "cranfield_loo_profile", "cranfield_random_rep"):
        for sp in SP.make_split(preset, data):
            assert sp.val.size > 0, f"{preset} fold {sp.fold} has no validation rows"


# --------------------------------------------------------------------------------------
# overlapping windows must not straddle a temporal cut
# --------------------------------------------------------------------------------------


def _overlapping(n=40, window_s=600.0, stride_s=300.0, start="2020-03-30"):
    """A window table with 50 % overlap, the shape MetroPT's window=60 stride=30 produces."""
    t0 = pd.Timestamp(start)
    t_end = (t0 + pd.to_timedelta(window_s + np.arange(n) * stride_s, unit="s")).to_numpy().astype("datetime64[ms]")
    t_start = t_end - np.timedelta64(int(window_s * 1000), "ms")
    return t_start, t_end


def test_temporal_split_purges_windows_that_straddle_a_cut():
    """Two windows 300 s apart share 300 s of raw samples. Placing them by window END alone
    puts the same observations either side of the fence, so the last validation window and the
    first test window are computed from partly the same data and calibration sees test input."""
    t_start, t_end = _overlapping()
    cut_a, cut_b = "2020-03-30T01:00:00", "2020-03-30T02:00:00"

    naive = SP.temporal_split(t_end, cut_a, cut_b, name="naive")
    a = np.datetime64(cut_a, "ms")
    assert (t_start[naive.test].min() < a) or (t_start[naive.val].min() < np.datetime64(cut_a, "ms"))

    purged = SP.temporal_split(t_end, cut_a, cut_b, name="purged", t_start=t_start)
    purged.check()
    assert purged.meta["purged_by_support"] and purged.meta["n_purged_straddling"] > 0
    # every surviving row's WHOLE support is on one side of every cut
    for part, lo, hi in (("train", None, a), ("val", a, np.datetime64(cut_b, "ms")), ("test", np.datetime64(cut_b, "ms"), None)):
        idx = getattr(purged, part)
        if lo is not None:
            assert (t_start[idx] >= lo).all(), f"{part} window starts before its own cut"
        if hi is not None:
            assert (t_end[idx] < hi).all(), f"{part} window ends after its own cut"
    # and the purge is exactly the straddling rows, not a wholesale drop
    kept = purged.train.size + purged.val.size + purged.test.size
    assert kept >= t_end.size - 4


def test_purged_parts_share_no_raw_support():
    t_start, t_end = _overlapping()
    sp = SP.temporal_split(t_end, "2020-03-30T01:00:00", "2020-03-30T02:00:00", t_start=t_start)
    spans = {p: [(t_start[i], t_end[i]) for i in getattr(sp, p)] for p in ("train", "val", "test")}
    for a_part, b_part in (("train", "val"), ("val", "test"), ("train", "test")):
        for s0, e0 in spans[a_part]:
            for s1, e1 in spans[b_part]:
                assert not (s0 < e1 and s1 < e0), f"{a_part}/{b_part} windows overlap in time"


def test_temporal_split_without_t_start_is_unchanged():
    """Point observations (one row = one instant) need no purge, and must not lose rows."""
    t = pd.date_range("2020-02-01", periods=120, freq="1D").to_numpy().astype("datetime64[ms]")
    sp = SP.temporal_split(t, "2020-04-01", "2020-04-11")
    sp.check(require_cover=True)
    assert sp.meta["n_purged_straddling"] == 0 and not sp.meta["purged_by_support"]


# --------------------------------------------------------------------------------------
# peer groups are placed atomically, so a cut cannot split one
# --------------------------------------------------------------------------------------


def _peer_frame(n=40, window_s=600.0, stride_s=600.0, start="2020-03-30"):
    """Rows in pairs: each pair is one peer group, the second member ending later."""
    t0 = pd.Timestamp(start)
    t_start = (t0 + pd.to_timedelta(np.repeat(np.arange(n // 2) * stride_s, 2), unit="s")).to_numpy().astype("datetime64[ms]")
    extra = np.tile([0.0, window_s], n // 2)  # the second peer of each pair ends much later
    t_end = t_start + (pd.to_timedelta(window_s + extra, unit="s").to_numpy().astype("timedelta64[ms]"))
    group = np.repeat(np.arange(n // 2), 2)
    return t_start, t_end, group


def test_a_temporal_cut_never_splits_a_peer_group():
    """Peer features are computed on the whole table before the split, so a cut falling inside
    a peer group is a real leak: the earlier member is a training row whose features already
    contain the later member, which is a test row."""
    from nebulax.bench.data import PEER_GROUP_COL

    t_start, t_end, group = _peer_frame()
    frame = pd.DataFrame({"train_id": "T01", "run_id": "r0", PEER_GROUP_COL: group})
    data = FakeData(t_end=t_end, unit=np.full(len(group), "T01", dtype=object),
                    group=np.asarray(group, dtype=object), labels=frame)
    data.t_start = t_start
    data.dataset = "sim"

    (sp,) = SP.make_split("temporal_fracs", data)
    sp.check()
    placed = {"train": set(group[sp.train]), "val": set(group[sp.val]), "test": set(group[sp.test])}
    for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
        assert not (placed[a] & placed[b]), f"peer group(s) {placed[a] & placed[b]} span {a}/{b}"
    assert sp.train.size and sp.test.size


def test_peer_group_widening_is_a_no_op_without_the_column():
    t_start, t_end, _ = _peer_frame()
    data = _fake(n=len(t_end))
    data.t_end = t_end
    data.t_start = t_start
    ts, te = SP._atomic_peer_supports(data, t_start, t_end)
    assert ts is t_start and te is t_end


def test_single_group_validation_fallback_carves_by_series_not_rows():
    """leave-one-group-out with only two groups: the remaining group is carved into
    validation by whole series (recordings), so no recording has windows on both sides."""
    from nebulax.bench import splits as SP
    groups = np.repeat(["p1", "p2"], 200).astype(object)
    series = np.repeat([f"rec{i}" for i in range(40)], 10).astype(object)  # 20 rows per recording... 10 each
    folds = SP.leave_one_group_out(groups, seed=0, val_frac=0.25, fine_groups=series)
    for sp in folds:
        assert sp.meta["val_carved_by"] == "series"
        assert sp.val.size > 0 and sp.train.size > 0
        assert not (set(series[sp.train]) & set(series[sp.val])), "a recording straddles train and val"
        assert not (set(series[sp.test]) & set(series[sp.val]))
    # without a fine grouping the old row fallback is still reachable and labelled
    folds = SP.leave_one_group_out(groups, seed=0, val_frac=0.25)
    assert all(sp.meta["val_carved_by"] == "row" for sp in folds)
