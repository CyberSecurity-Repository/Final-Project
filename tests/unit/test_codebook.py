"""Tests for the display-only value codebook.

Every label is transcribed verbatim from the dataset description
(`docs/reference/e-shop_clothing_2008_data_description.txt`). These tests pin the
exact wording, prove the maps fully cover the observed/contract domains, and
confirm decoding never raises and never enters validation.
"""

from __future__ import annotations

from pathlib import Path

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.reference import codebook as cb
from retail_clickstream_ai.validation.contract import DatasetContract

CODEBOOK_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "analyst" / "codebook.json"


def test_exact_labels_verbatim() -> None:
    assert cb.MAIN_CATEGORY == {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}
    assert cb.COUNTRY[12] == "unidentified"
    assert cb.COUNTRY[42] == "USA"
    assert cb.COUNTRY[47] == "org (*.org)"
    assert cb.COLOUR[9] == "of many colors"
    assert cb.COLOUR[14] == "white"
    assert cb.LOCATION[6] == "bottom right"
    assert cb.MODEL_PHOTOGRAPHY == {1: "en face", 2: "profile"}
    assert cb.PRICE_2 == {1: "yes", 2: "no"}


def test_maps_fully_cover_observed_domains() -> None:
    assert set(cb.MAIN_CATEGORY) == {1, 2, 3, 4}
    assert set(cb.COUNTRY) == set(range(1, 48))  # observed 1–47, contiguous
    assert set(cb.COLOUR) == set(range(1, 15))
    assert set(cb.LOCATION) == set(range(1, 7))
    assert set(cb.MODEL_PHOTOGRAPHY) == {1, 2}
    assert set(cb.PRICE_2) == {1, 2}


def test_codebook_matches_contract_allowed_values() -> None:
    """Decodable columns must cover exactly the contract's allowed set."""
    allowed = {
        c.name: set(c.allowed_values)
        for c in DatasetContract.build().columns
        if c.allowed_values is not None
    }
    for column in (d.MAIN_CATEGORY, d.COLOUR, d.LOCATION, d.MODEL_PHOTOGRAPHY, d.PRICE_2):
        assert set(cb.CODEBOOK[column]) == allowed[column]


def test_has_labels() -> None:
    assert cb.has_labels(d.MAIN_CATEGORY)
    assert cb.has_labels(d.COUNTRY)
    # No label table in the source for these -> shown as raw numbers.
    assert not cb.has_labels(d.PRICE)
    assert not cb.has_labels(d.PAGE)
    assert not cb.has_labels(d.CLOTHING_MODEL)


def test_decode_and_fallbacks() -> None:
    assert cb.decode(d.MAIN_CATEGORY, 1) == "trousers"
    assert cb.decode(d.PRICE_2, "2") == "no"  # string code is coerced
    assert cb.decode(d.MAIN_CATEGORY, 99) == "99"  # unmapped code -> str
    assert cb.decode(d.PRICE, 55) == "55"  # column without labels -> str
    assert cb.decode(d.COLOUR, "beige") == "beige"  # non-int -> str fallback


def test_label_with_code() -> None:
    assert cb.label_with_code(d.MAIN_CATEGORY, 4) == "4 (sale)"
    assert cb.label_with_code(d.PRICE, 55) == "55"  # no labels -> bare code


def test_codebook_serialization_is_stable() -> None:
    assert cb.to_json() == cb.to_json()


def test_committed_codebook_matches_build() -> None:
    assert CODEBOOK_ARTIFACT.exists(), (
        "Missing artifact — regenerate with: "
        "python -m retail_clickstream_ai.pipeline.data build-codebook"
    )
    assert CODEBOOK_ARTIFACT.read_text(encoding="utf-8") == cb.to_json()
