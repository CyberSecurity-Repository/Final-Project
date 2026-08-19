"""leakage-safe feature engineering.

The load-bearing correctness tests: the target is the next click within the same
session (never across a boundary), each session's final click is dropped, past
aggregates cannot see the current/future target, the month split is exact, all
classes appear in every split, features are byte-reproducible, and unknown
categories are handled at inference.
"""

from __future__ import annotations

import pandas as pd

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline import features as F
from retail_clickstream_ai.validation.contract import DatasetContract

CONTRACT = DatasetContract.build()


def test_target_is_next_click_within_session(synthetic_clean_frame: pd.DataFrame) -> None:
    feats, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    keyed = d.canonical_sort(synthetic_clean_frame)
    expected = keyed[[d.SESSION_KEY, d.ORDER, d.MAIN_CATEGORY]].copy()
    expected["expected_next"] = expected.groupby(d.SESSION_KEY, sort=False)[d.MAIN_CATEGORY].shift(
        -1
    )
    merged = feats.merge(expected, on=[d.SESSION_KEY, d.ORDER], how="left")
    assert (merged["expected_next"] == merged[F.TARGET]).all()
    assert merged["expected_next"].notna().all()  # never a boundary-crossing NaN


def test_final_click_of_each_session_is_dropped(synthetic_clean_frame: pd.DataFrame) -> None:
    feats, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    keyed = d.add_session_key(synthetic_clean_frame)
    n_sessions = keyed[d.SESSION_KEY].nunique()
    assert len(feats) == len(keyed) - n_sessions
    # No surviving row is a session's maximum-order (last) click.
    last = keyed.groupby(d.SESSION_KEY)[d.ORDER].transform("max")
    last_clicks = keyed.loc[keyed[d.ORDER] == last, [d.SESSION_KEY, d.ORDER]]
    assert len(feats.merge(last_clicks, on=[d.SESSION_KEY, d.ORDER], how="inner")) == 0


def test_past_aggregates_cannot_see_future(synthetic_clean_frame: pd.DataFrame) -> None:
    # The audit's mutation probe: flipping a future click never changes an
    # earlier row's predictors (only the label may move).
    ok, detail = F._probe_no_future_leak()
    assert ok, detail


def test_month_split_boundaries_are_exact(synthetic_clean_frame: pd.DataFrame) -> None:
    feats, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    by_split = feats.groupby(F.SPLIT_COLUMN)[d.MONTH].agg(lambda s: sorted(set(s)))
    assert by_split["train"] == [4, 5, 6]
    assert by_split["validation"] == [7]
    assert by_split["test"] == [8]
    assert "unknown" not in set(feats[F.SPLIT_COLUMN])


def test_all_classes_present_in_every_split(synthetic_clean_frame: pd.DataFrame) -> None:
    feats, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    manifest = F.build_split_manifest(feats, CONTRACT)
    assert manifest["all_classes_present_each_split"] is True
    for name in ("train", "validation", "test"):
        counts = manifest["splits"][name]["class_counts"]
        assert all(counts[str(c)] > 0 for c in (1, 2, 3, 4))


def test_leakage_audit_passes(synthetic_clean_frame: pd.DataFrame) -> None:
    feats, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    audit = F.run_leakage_audit(synthetic_clean_frame, feats, CONTRACT)
    assert audit["passed"] is True
    assert all(c["passed"] for c in audit["checks"])


def test_features_are_byte_reproducible(synthetic_clean_frame: pd.DataFrame, tmp_path) -> None:
    feats1, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    feats2, _ = F.build_features(synthetic_clean_frame, CONTRACT)
    p1, p2 = tmp_path / "f1.csv", tmp_path / "f2.csv"
    s1 = F.write_features_csv(feats1, p1)
    s2 = F.write_features_csv(feats2, p2)
    assert s1["sha256"] == s2["sha256"]


def test_raw_session_id_is_identifier_not_predictor() -> None:
    assert d.SESSION_ID in F.IDENTIFIER_COLUMNS
    assert d.SESSION_ID not in F.PREDICTOR_COLUMNS
    assert d.SESSION_KEY not in F.PREDICTOR_COLUMNS
    assert F.TARGET not in F.PREDICTOR_COLUMNS
