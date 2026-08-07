"""The machine-validatable dataset contract.

This module is the single source of truth for what the raw clickstream dataset
*is*, expressed as data that both a human and a validator can read. The contract
is **generated deterministically** from verified constants (no timestamps, no
environment-dependent values), so serializing it twice yields byte-identical
output and the committed ``dataset_contract.json`` artifact stays reproducible.

Every constraint below was measured from the real file during Stage 2 inspection
(SHA-256 ``fcc167bb…8115d``); publisher-documented meanings (e.g. category label
text) are recorded as descriptions, never as validator constraints, so the
anti-hallucination boundary holds: constraints come from observed bytes, prose
from the publisher is clearly labeled as documented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retail_clickstream_ai.pipeline import data as d

CONTRACT_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Verified raw-file facts (measured from the bytes during Stage 2).
# --------------------------------------------------------------------------- #
RAW_SHA256 = "fcc167bbd0badd4c9685bd8543097e318f8228e48075335db7cd781cee88115d"
RAW_SIZE_BYTES = 6675312
RAW_ROW_COUNT = 165474
RAW_COLUMN_COUNT = 14
OBSERVED_SESSION_COUNT = 24026
OBSERVED_MONTHS = (4, 5, 6, 7, 8)

# Chronological, session-safe split (never a random row split).
TRAIN_MONTHS = (4, 5, 6)
VALIDATION_MONTHS = (7,)
TEST_MONTHS = (8,)

_PAGE2_PATTERN = r"^[A-P][0-9]+$"


@dataclass(frozen=True)
class ColumnSpec:
    """One column's contract: identity, type, nullability, and its constraint.

    A column carries at most one of ``allowed_values`` / ``value_range`` /
    ``pattern``. ``description`` may mix observed facts with clearly-labeled
    publisher documentation, but only the typed constraint is enforced.
    """

    name: str
    source_name: str
    dtype: str
    nullable: bool
    description: str
    allowed_values: tuple[int, ...] | None = None
    value_range: tuple[int | None, int | None] | None = None
    pattern: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "source_name": self.source_name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "description": self.description,
        }
        if self.allowed_values is not None:
            payload["allowed_values"] = list(self.allowed_values)
        if self.value_range is not None:
            payload["range"] = {"min": self.value_range[0], "max": self.value_range[1]}
        if self.pattern is not None:
            payload["pattern"] = self.pattern
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ColumnSpec:
        rng = payload.get("range")
        return cls(
            name=payload["name"],
            source_name=payload["source_name"],
            dtype=payload["dtype"],
            nullable=payload["nullable"],
            description=payload["description"],
            allowed_values=(
                tuple(payload["allowed_values"]) if "allowed_values" in payload else None
            ),
            value_range=(rng["min"], rng["max"]) if rng is not None else None,
            pattern=payload.get("pattern"),
        )


# --------------------------------------------------------------------------- #
# The normalized schema — order matches the source header.
# Constraints are the OBSERVED domains; label text is DOCUMENTED (publisher).
# --------------------------------------------------------------------------- #
_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        d.YEAR,
        "year",
        "int64",
        False,
        "Calendar year of the click. Observed: constant 2008.",
        value_range=(2008, 2008),
    ),
    ColumnSpec(
        d.MONTH,
        "month",
        "int64",
        False,
        "Month of the click. Documented + observed: 4=April … 8=August.",
        allowed_values=(4, 5, 6, 7, 8),
    ),
    ColumnSpec(
        d.DAY,
        "day",
        "int64",
        False,
        "Day of the month (observed 1–31).",
        value_range=(1, 31),
    ),
    ColumnSpec(
        d.ORDER,
        "order",
        "int64",
        False,
        "Click sequence within a session, starting at 1 (documented 'order'). "
        "Observed: a contiguous 1..n run per verified session.",
        value_range=(1, None),
    ),
    ColumnSpec(
        d.COUNTRY,
        "country",
        "int64",
        False,
        "Coded country of origin (documented, IP-based). Observed domain 1–47.",
        value_range=(1, 47),
    ),
    ColumnSpec(
        d.SESSION_ID,
        "session ID",
        "int64",
        False,
        "Per-session id (documented 'short record'). Observed 1–24026 and, in "
        "this file, globally unique; the canonical session key still composes it "
        "with the date (see session_key) so it stays correct if ids reset per day.",
        value_range=(1, None),
    ),
    ColumnSpec(
        d.MAIN_CATEGORY,
        "page 1 (main category)",
        "int64",
        False,
        "Main product category. Documented codes: 1=trousers, 2=skirts, "
        "3=blouses, 4=sale. This is the source column for the prediction target.",
        allowed_values=(1, 2, 3, 4),
    ),
    ColumnSpec(
        d.CLOTHING_MODEL,
        "page 2 (clothing model)",
        "string",
        False,
        "Product code (documented: 217 products), e.g. 'A13'. Observed pattern "
        "letter+digits; first letter (A/B/C/P) aligns with the main category.",
        pattern=_PAGE2_PATTERN,
    ),
    ColumnSpec(
        d.COLOUR,
        "colour",
        "int64",
        False,
        "Coded product colour. Documented 14-colour palette; observed domain 1–14.",
        allowed_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    ),
    ColumnSpec(
        d.LOCATION,
        "location",
        "int64",
        False,
        "Photo location on the page (documented: screen split into six regions). "
        "Observed domain 1–6.",
        allowed_values=(1, 2, 3, 4, 5, 6),
    ),
    ColumnSpec(
        d.MODEL_PHOTOGRAPHY,
        "model photography",
        "int64",
        False,
        "Model pose. Documented: 1=en face, 2=profile. Observed domain 1–2.",
        allowed_values=(1, 2),
    ),
    ColumnSpec(
        d.PRICE,
        "price",
        "int64",
        False,
        "Price in US dollars (documented). Observed integers 18–82; validated as "
        "a positive integer rather than a fixed set so it stays feature-friendly.",
        value_range=(1, None),
    ),
    ColumnSpec(
        d.PRICE_2,
        "price 2",
        "int64",
        False,
        "Whether the price is above the category average. Documented: 1=yes, "
        "2=no. Observed domain 1–2.",
        allowed_values=(1, 2),
    ),
    ColumnSpec(
        d.PAGE,
        "page",
        "int64",
        False,
        "Page number within the e-store (documented 1–5). Observed domain 1–5.",
        allowed_values=(1, 2, 3, 4, 5),
    ),
)

_CLEANING_RULES: tuple[str, ...] = (
    f"Parse with delimiter '{d.RAW_DELIMITER}' and UTF-8/ASCII encoding; the file "
    "uses CRLF line endings.",
    "Normalize column names with normalize_column_name (lowercase, non-alphanumeric "
    "runs -> single underscore, trimmed).",
    "Keep 'page 2 (clothing model)' as text; parse all other columns as int64.",
    "Require zero nulls in every column (the source has none).",
    "Reject full-row duplicates (the source has none).",
    "Derive session_key = 'YYYY-MM-DD-s<session_id>' as the canonical session id.",
    "Sort deterministically by (year, month, day, session_id, order) ascending "
    "(stable). This sort defines click order regardless of on-disk row order.",
    "Require each session's 'order' values to form a contiguous 1..n sequence "
    "(no gaps, no duplicates).",
)

_ASSUMPTIONS: tuple[str, ...] = (
    "Data covers April–August 2008 for one online clothing shop; it is a workflow "
    "demonstration and NOT evidence of present-day shopping behavior.",
    "In this file session_id is globally unique, but the contract does not assume "
    "that: the composite session_key is authoritative.",
    "Coded columns (country, colour, location, etc.) are categorical ids, not "
    "quantities; only price/order/day are treated as ordered numbers.",
    "Publisher label meanings (category/colour/pose text) are documented, not "
    "independently verifiable from the numeric file, and are recorded as such.",
)

_CONSTRAINTS: tuple[str, ...] = (
    "A random row split is forbidden: clicks in a session are correlated and would "
    "leak across splits. Use the chronological month split below.",
    "Features must use only the current and earlier clicks of the same session; no "
    "value derived from click t+1 or later may enter the features (target excepted).",
    "All four main categories must be present in every split (verified: they are).",
)


@dataclass
class DatasetContract:
    """A structured, JSON-backed dataset contract.

    Wraps the generated contract ``data`` and offers typed views the validator
    needs. Build it deterministically with :meth:`build`; persist it with
    :meth:`write`; reload it with :meth:`load`.
    """

    data: dict[str, Any]

    # -- construction ------------------------------------------------------- #
    @classmethod
    def build(cls) -> DatasetContract:
        """Generate the contract from verified constants (deterministic)."""
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "dataset": {
                "name": "Clickstream Data for Online Shopping",
                "source_urls": {
                    "uci": "https://archive.ics.uci.edu/dataset/553/"
                    "clickstream+data+for+online+shopping",
                    "kaggle": "https://www.kaggle.com/datasets/tunguz/"
                    "clickstream-data-for-online-shopping",
                },
                "kaggle_slug": "tunguz/clickstream-data-for-online-shopping",
                "publisher": "Mariusz Łapczyński and Sylwester Białowąs "
                "(UCI Machine Learning Repository, dataset 553)",
                "license": "CC BY 4.0",
                "documented_time_period": "April–August 2008 (five months)",
                "raw_file": {
                    "name": d.RAW_FILENAME,
                    "sha256": RAW_SHA256,
                    "size_bytes": RAW_SIZE_BYTES,
                    "delimiter": d.RAW_DELIMITER,
                    "encoding": d.RAW_ENCODING,
                    "line_ending": "CRLF",
                    "row_count": RAW_ROW_COUNT,
                    "column_count": RAW_COLUMN_COUNT,
                    "committed": False,
                },
            },
            "schema": [c.to_dict() for c in _SCHEMA],
            "session_key": {
                "derived_column": d.SESSION_KEY,
                "columns": list(d.SESSION_KEY_COLUMNS),
                "format": "YYYY-MM-DD-s<session_id>",
                "observed_session_count": OBSERVED_SESSION_COUNT,
                "session_id_globally_unique_in_file": True,
                "rationale": "Composite key is canonical so sessions stay distinct "
                "even if a bare session_id is reused on another day.",
            },
            "sort": {
                "columns": list(d.SORT_COLUMNS),
                "order": "ascending",
                "policy": "Click order within a session is defined by 'order'; rows "
                "are re-sorted deterministically regardless of on-disk order.",
            },
            "cleaning_rules": list(_CLEANING_RULES),
            "target": {
                "name": "next_main_category",
                "definition": "Within the same verified session (sorted by 'order' "
                "ascending), the next click's page_1_main_category (shift(-1)).",
                "source_column": d.MAIN_CATEGORY,
                "classes": [1, 2, 3, 4],
                "drop_last_click_per_session": True,
                "rows_total": RAW_ROW_COUNT,
                "rows_with_label": RAW_ROW_COUNT - OBSERVED_SESSION_COUNT,
            },
            "split": {
                "strategy": "chronological month-based (never random)",
                "train_months": list(TRAIN_MONTHS),
                "validation_months": list(VALIDATION_MONTHS),
                "test_months": list(TEST_MONTHS),
                "rationale": "Sessions are correlated; a random row split leaks "
                "future clicks of a session into training. See "
                "docs/decisions/0001-no-random-row-split.md.",
            },
            "required_output_columns": list(d.REQUIRED_OUTPUT_COLUMNS),
            "assumptions": list(_ASSUMPTIONS),
            "constraints": list(_CONSTRAINTS),
        }
        return cls(payload)

    @classmethod
    def load(cls, path: str | Path) -> DatasetContract:
        """Load a contract from a JSON file."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- serialization ------------------------------------------------------ #
    def to_json(self) -> str:
        """Serialize to a stable JSON string (sorted keys, trailing newline)."""
        return json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to ``path`` and return the path."""
        out = Path(path)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    # -- typed views used by the validator ---------------------------------- #
    @property
    def columns(self) -> list[ColumnSpec]:
        return [ColumnSpec.from_dict(c) for c in self.data["schema"]]

    @property
    def raw_sha256(self) -> str:
        return str(self.data["dataset"]["raw_file"]["sha256"])

    @property
    def session_key_columns(self) -> list[str]:
        return list(self.data["session_key"]["columns"])

    @property
    def sort_columns(self) -> list[str]:
        return list(self.data["sort"]["columns"])

    @property
    def required_months(self) -> list[int]:
        split = self.data["split"]
        return [
            *split["train_months"],
            *split["validation_months"],
            *split["test_months"],
        ]
