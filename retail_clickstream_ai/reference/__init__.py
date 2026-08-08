"""Reference lookups derived from the dataset's own documentation.

Currently holds the display **codebook** (integer code -> human-readable label),
transcribed verbatim from the authoritative dataset description. These are
presentation-only labels; the pipeline and models always use the canonical
integer codes.
"""

from __future__ import annotations

from retail_clickstream_ai.reference.codebook import (
    CODEBOOK,
    build_codebook,
    decode,
    has_labels,
    label_with_code,
)

__all__ = [
    "CODEBOOK",
    "build_codebook",
    "decode",
    "has_labels",
    "label_with_code",
]
