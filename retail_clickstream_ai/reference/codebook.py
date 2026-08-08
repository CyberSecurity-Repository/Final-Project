"""Human-readable value labels — DISPLAY ONLY.

Every mapping below is transcribed **verbatim** from the authoritative dataset
description, committed at
``docs/reference/e-shop_clothing_2008_data_description.txt`` (UCI ML Repository
dataset #553; Łapczyński M., Białowąs S. (2013); CC BY 4.0). Nothing here is
invented — if the description does not name a code, it is not decoded (e.g. the
217 product codes in ``page_2_clothing_model`` have no names in the source).

**Scope boundary:** these labels exist so the running app can explain a result
("trousers" instead of ``1``). They are **never** validation constraints and
**never** enter feature or model logic — the pipeline keeps the canonical
integer codes everywhere. Decoding happens only at the presentation edge.
"""

from __future__ import annotations

import json
from pathlib import Path

from retail_clickstream_ai.pipeline import data as d

SOURCE = (
    "e-shop clothing 2008 data description.txt — UCI ML Repository dataset #553; "
    "Łapczyński M., Białowąs S. (2013); CC BY 4.0"
)

# --- code -> label maps (verbatim from the description) -------------------- #
MAIN_CATEGORY: dict[int, str] = {
    1: "trousers",
    2: "skirts",
    3: "blouses",
    4: "sale",
}

COUNTRY: dict[int, str] = {
    1: "Australia",
    2: "Austria",
    3: "Belgium",
    4: "British Virgin Islands",
    5: "Cayman Islands",
    6: "Christmas Island",
    7: "Croatia",
    8: "Cyprus",
    9: "Czech Republic",
    10: "Denmark",
    11: "Estonia",
    12: "unidentified",
    13: "Faroe Islands",
    14: "Finland",
    15: "France",
    16: "Germany",
    17: "Greece",
    18: "Hungary",
    19: "Iceland",
    20: "India",
    21: "Ireland",
    22: "Italy",
    23: "Latvia",
    24: "Lithuania",
    25: "Luxembourg",
    26: "Mexico",
    27: "Netherlands",
    28: "Norway",
    29: "Poland",
    30: "Portugal",
    31: "Romania",
    32: "Russia",
    33: "San Marino",
    34: "Slovakia",
    35: "Slovenia",
    36: "Spain",
    37: "Sweden",
    38: "Switzerland",
    39: "Ukraine",
    40: "United Arab Emirates",
    41: "United Kingdom",
    42: "USA",
    43: "biz (*.biz)",
    44: "com (*.com)",
    45: "int (*.int)",
    46: "net (*.net)",
    47: "org (*.org)",
}

COLOUR: dict[int, str] = {
    1: "beige",
    2: "black",
    3: "blue",
    4: "brown",
    5: "burgundy",
    6: "gray",
    7: "green",
    8: "navy blue",
    9: "of many colors",
    10: "olive",
    11: "pink",
    12: "red",
    13: "violet",
    14: "white",
}

LOCATION: dict[int, str] = {
    1: "top left",
    2: "top in the middle",
    3: "top right",
    4: "bottom left",
    5: "bottom in the middle",
    6: "bottom right",
}

MODEL_PHOTOGRAPHY: dict[int, str] = {
    1: "en face",
    2: "profile",
}

# PRICE 2 answers "is the price higher than the category average?" -> yes / no.
PRICE_2: dict[int, str] = {
    1: "yes",
    2: "no",
}

# Column-level meaning text (verbatim phrasing from the description), for tooltips.
COLUMN_MEANINGS: dict[str, str] = {
    d.MAIN_CATEGORY: "main product category",
    d.COUNTRY: "country of origin of the IP address",
    d.COLOUR: "colour of product",
    d.LOCATION: "photo location on the page (the screen is divided into six parts)",
    d.MODEL_PHOTOGRAPHY: "model photography",
    d.PRICE_2: (
        "whether the price of a particular product is higher than the average "
        "price for the entire product category"
    ),
}

# The single lookup the app uses: normalized column name -> code -> label.
CODEBOOK: dict[str, dict[int, str]] = {
    d.MAIN_CATEGORY: MAIN_CATEGORY,
    d.COUNTRY: COUNTRY,
    d.COLOUR: COLOUR,
    d.LOCATION: LOCATION,
    d.MODEL_PHOTOGRAPHY: MODEL_PHOTOGRAPHY,
    d.PRICE_2: PRICE_2,
}


def has_labels(column: str) -> bool:
    """True when ``column`` has a decode table (else the raw number is shown)."""
    return column in CODEBOOK


def decode(column: str, code: int | str) -> str:
    """Return the label for ``code`` in ``column``, or ``str(code)`` as fallback.

    Never raises: an unknown column, an unmapped code, or a non-integer input
    all degrade gracefully to the string form of the code.
    """
    labels = CODEBOOK.get(column)
    if labels is None:
        return str(code)
    try:
        key = int(code)
    except (TypeError, ValueError):
        return str(code)
    return labels.get(key, str(code))


def label_with_code(column: str, code: int | str) -> str:
    """Render ``"1 (trousers)"`` for decodable columns; ``"1"`` otherwise."""
    if not has_labels(column):
        return str(code)
    return f"{code} ({decode(column, code)})"


def build_codebook() -> dict[str, object]:
    """Assemble the JSON-serializable codebook (deterministic)."""
    columns: dict[str, object] = {}
    for column, labels in CODEBOOK.items():
        columns[column] = {
            "meaning": COLUMN_MEANINGS.get(column, ""),
            "values": {str(code): label for code, label in labels.items()},
        }
    return {
        "source": SOURCE,
        "note": (
            "Display-only labels transcribed verbatim from the dataset "
            "description. Not a validation constraint; the pipeline uses the "
            "integer codes. Product codes (page_2_clothing_model) have no names "
            "in the source and are shown as-is."
        ),
        "columns": columns,
    }


def to_json() -> str:
    """Serialize the codebook to a stable JSON string (sorted keys + newline)."""
    return json.dumps(build_codebook(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write(path: str | Path) -> Path:
    """Write :func:`to_json` to ``path`` and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(), encoding="utf-8")
    return out
