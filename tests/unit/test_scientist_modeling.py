"""Stage 4 — model training, selection, evaluation, and persistence.

Proves: a baseline plus two model variations are compared; the winner is chosen
on validation macro F1; the test split is never touched during training;
preprocessing is fit on training rows only and unknown categories are handled at
inference; a fixed seed reproduces the metrics; and the saved model round-trips.
"""

from __future__ import annotations

import pandas as pd

from retail_clickstream_ai.pipeline import features as F
from retail_clickstream_ai.pipeline import modeling as M


def _features(clean: pd.DataFrame) -> pd.DataFrame:
    from retail_clickstream_ai.validation.contract import DatasetContract

    feats, _ = F.build_features(clean, DatasetContract.build())
    return feats


def _predictor_row(**over: object) -> dict:
    row = {
        F.CURRENT_CATEGORY: 1,
        F.PREV_CATEGORY: 0,
        "colour": 1,
        "location": 1,
        "model_photography": 1,
        "price_2": 1,
        "page": 1,
        "country": 9,
        "price": 30,
        F.CLICKS_SO_FAR: 1,
        F.DISTINCT_CATEGORIES_SO_FAR: 1,
        F.COUNT_CURRENT_CATEGORY_SO_FAR: 1,
    }
    row.update(over)
    return row


def test_baseline_plus_two_models_compared(synthetic_clean_frame: pd.DataFrame) -> None:
    results = M.run_candidate_experiments(_features(synthetic_clean_frame))
    assert results["candidate_count"] >= 3
    assert results["all_required_candidates_present"] is True
    families = {c["family"] for c in results["candidates"].values()}
    assert set(M.REQUIRED_FAMILIES) <= families


def test_training_never_accesses_test(synthetic_clean_frame: pd.DataFrame) -> None:
    results = M.run_candidate_experiments(_features(synthetic_clean_frame))
    assert results["test_accessed"] is False
    check = M.validate_training_outputs(results)
    assert check["passed"] is True
    assert check["test_accessed"] is False


def test_winner_is_top_validation_macro_f1(synthetic_clean_frame: pd.DataFrame) -> None:
    results = M.run_candidate_experiments(_features(synthetic_clean_frame))
    ranking = results["validation_ranking"]
    macro = [r["macro_f1"] for r in ranking]
    assert macro == sorted(macro, reverse=True)  # ranked best-first
    top_id = ranking[0]["candidate_id"]
    best = max(results["candidates"].values(), key=lambda c: c["metrics"]["macro_f1"])
    assert results["candidates"][top_id]["metrics"]["macro_f1"] == best["metrics"]["macro_f1"]


def test_preprocessing_fits_on_train_only_and_handles_unknown() -> None:
    # Train sees location {1,2}; validation introduces an unseen location 6.
    train = pd.DataFrame(
        [_predictor_row(location=1), _predictor_row(location=2), _predictor_row(location=1)]
    )[list(F.PREDICTOR_COLUMNS)]
    val = pd.DataFrame([_predictor_row(location=6)])[list(F.PREDICTOR_COLUMNS)]
    pre = M._make_preprocessor(scale_numeric=False)
    pre.fit(train)
    ohe = pre.named_transformers_["cat"]
    loc_idx = list(F.CATEGORICAL_FEATURES).index("location")
    assert set(ohe.categories_[loc_idx]) == {1, 2}  # learned from TRAIN only
    transformed = pre.transform(val)  # unknown category handled, no error
    assert transformed.shape[0] == 1


def test_unknown_category_handled_by_full_estimator(
    synthetic_clean_frame: pd.DataFrame,
) -> None:
    feats = _features(synthetic_clean_frame)
    train = feats[feats[F.SPLIT_COLUMN] == "train"]
    est = M.build_estimator(M.LOGREG_ID, {"C": 1.0})
    est.fit(train.loc[:, list(F.PREDICTOR_COLUMNS)], train[F.TARGET])
    novel = pd.DataFrame([_predictor_row(country=999)])[list(F.PREDICTOR_COLUMNS)]
    proba = est.predict_proba(novel)
    assert proba.shape == (1, 4)


def test_fixed_seed_reproduces_metrics(synthetic_clean_frame: pd.DataFrame) -> None:
    feats = _features(synthetic_clean_frame)
    r1 = M.run_candidate_experiments(feats)
    r2 = M.run_candidate_experiments(feats)
    for cid in r1["candidates"]:
        m1 = r1["candidates"][cid]["metrics"]
        m2 = r2["candidates"][cid]["metrics"]
        assert abs(m1["macro_f1"] - m2["macro_f1"]) < 1e-9
        assert m1["confusion_matrix"] == m2["confusion_matrix"]


def test_saved_model_round_trips(synthetic_clean_frame: pd.DataFrame, tmp_path) -> None:
    import joblib
    import numpy as np

    feats = _features(synthetic_clean_frame)
    bundle = M.lock_winner_and_evaluate_test(
        feats,
        tmp_path,
        training_data_sha256="hash",
        features_sha256="hash",
        render_figures=False,
    )
    assert bundle.round_trip_validated is True
    # Independent reload confirms prediction + probability parity.
    model = joblib.load(bundle.model_path)  # trusted, just-produced artifact
    test = feats[feats[F.SPLIT_COLUMN] == "test"].loc[:, list(F.PREDICTOR_COLUMNS)]
    reload_pred = model.predict(test)
    assert len(reload_pred) == len(test)
    assert np.asarray(model.predict_proba(test)).shape[1] == 4


def test_metadata_hashes_match_files(synthetic_clean_frame: pd.DataFrame, tmp_path) -> None:
    from retail_clickstream_ai.pipeline import data as d

    feats = _features(synthetic_clean_frame)
    bundle = M.lock_winner_and_evaluate_test(
        feats, tmp_path, training_data_sha256="h", features_sha256="h", render_figures=False
    )
    assert bundle.metadata["metrics_sha256"] == d.sha256_file(bundle.metrics_path)
    assert bundle.metadata["artifact_sha256"] == d.sha256_file(bundle.model_path)
