"""Independent test oracle.

This module deliberately imports *nothing* from the rankfusion package: the
expected ranking is recomputed from the static complete catalog file using
only the Python standard library. If the product and this oracle ever share
a bug, tests cannot catch it — so they intentionally do not share code.

The universe is the union of the source candidate sets (never augmented with
all-zero unknown ids). Missing components score exactly 0. Ranking is exact
weighted sum descending, then item_id Unicode code-point ascending.
"""
from __future__ import annotations

import fractions
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


def _parse_fraction(text: str) -> Fraction:
    text = str(text).strip()
    if "/" in text:
        num, den = text.split("/")
        return Fraction(int(num), int(den))
    return Fraction(text)


def load_expected(catalog_path: str | Path, weights: dict[str, str], k: int) -> dict[str, Any]:
    """Compute the exact expected answer straight from the catalog file."""
    data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    parsed_weights = {sid: _parse_fraction(w) for sid, w in weights.items()}

    universe: dict[str, dict[str, Fraction]] = {}
    per_source: dict[str, dict[str, Fraction]] = {}
    for source in data["sources"]:
        sid = source["source_id"]
        if sid not in parsed_weights:
            raise ValueError(f"oracle: no weight supplied for source {sid!r}")
        scores: dict[str, Fraction] = {}
        for entry in source["entries"]:
            score = _parse_fraction(entry["score"])
            if not (0 <= score <= 1):
                raise ValueError("oracle: score outside [0, 1]")
            scores[entry["item_id"]] = score
            universe.setdefault(entry["item_id"], {})
        per_source[sid] = scores

    totals: list[tuple[str, Fraction, dict[str, Fraction]]] = []
    for item_id in sorted(universe):
        components: dict[str, Fraction] = {}
        total = Fraction(0)
        for sid, weight in parsed_weights.items():
            score = per_source.get(sid, {}).get(item_id, Fraction(0))
            components[sid] = score
            total += weight * score
        totals.append((item_id, total, components))

    # Exact ordering: total desc, item_id asc. Fraction comparisons are exact.
    totals.sort(key=lambda row: (-row[1], row[0]))
    chosen = totals[: min(k, len(totals))]
    return {
        "universe_size": len(totals),
        "order": [
            {
                "item_id": item_id,
                "total": str(total),
                "components": {sid: str(score) for sid, score in components.items()},
            }
            for item_id, total, components in totals
        ],
        "top_k": [
            {
                "item_id": item_id,
                "total": str(total),
                "components": {sid: str(score) for sid, score in components.items()},
            }
            for item_id, total, components in chosen
        ],
    }


def float_ordering_differs(catalog_path: str | Path, weights: dict[str, str], k: int) -> bool:
    """Guard used by the near-tie scenario: confirm float arithmetic would
    actually conflate two totals that exact arithmetic keeps apart."""
    expected = load_expected(catalog_path, weights, k)
    float_totals = [
        (row["item_id"], _float_total(row["components"], weights)) for row in expected["order"]
    ]
    exact_ids = [row["item_id"] for row in expected["order"]]
    float_ranked = sorted(float_totals, key=lambda row: (-row[1], row[0]))
    float_ids = [item_id for item_id, _ in float_ranked]
    return exact_ids != float_ids


def _float_total(components: dict[str, str], weights: dict[str, str]) -> float:
    total = 0.0
    for sid, score_text in components.items():
        total += float(_parse_fraction(score_text)) * float(_parse_fraction(weights[sid]))
    return total


assert fractions  # re-export guard: keeps Fraction dependency explicit
