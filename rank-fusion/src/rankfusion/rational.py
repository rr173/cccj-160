"""Exact rational arithmetic helpers.

Scores and weights are *never* converted to float anywhere in this project.
They travel over the wire as JSON strings (e.g. ``"3/5"``, ``"0.25"``) and
are parsed here into :class:`fractions.Fraction` instances.
"""
from __future__ import annotations

import fractions
import re
from fractions import Fraction

# A string is accepted as an exact rational if it is a common fraction "a/b",
# a decimal "0.75" or an integer "2". No scientific notation: the contract
# only allows exact rational strings.
_DECIMAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def parse_rational(value: object, *, field: str = "value") -> Fraction:
    """Parse an exact rational value into a :class:`Fraction`.

    Accepts:
      * :class:`fractions.Fraction` (returned unchanged),
      * :class:`int`,
      * strings like ``"3/5"``, ``"-2/7"``, ``"0.25"``, ``"1"``.

    Floats are rejected explicitly — converting them would silently lose
    precision and violate the "no float" contract.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):  # bool is a subclass of int; reject it.
        raise ValueError(f"{field}: boolean is not a valid rational")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise ValueError(
            f"{field}: float value {value!r} is forbidden; pass an exact "
            f"rational string such as '3/5' or '0.25'"
        )
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected exact rational string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field}: empty rational string")
    if "/" in text:
        parts = text.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{field}: malformed fraction {value!r}")
        num_s, den_s = parts
        if not (re.fullmatch(r"[+-]?\d+", num_s) and re.fullmatch(r"\d+", den_s)):
            raise ValueError(f"{field}: malformed fraction {value!r}")
        if int(den_s) == 0:
            raise ValueError(f"{field}: zero denominator in {value!r}")
        return Fraction(int(num_s), int(den_s))
    if _DECIMAL_RE.match(text):
        return Fraction(text)
    raise ValueError(f"{field}: unparseable rational {value!r}")


def parse_score(value: object, *, field: str = "score") -> Fraction:
    """Parse a catalog score and verify it lies in the closed interval [0, 1]."""
    score = parse_rational(value, field=field)
    if score < 0 or score > 1:
        raise ValueError(f"{field}: score {value!r} is outside [0, 1]")
    return score


def parse_weight(value: object, *, source_id: str) -> Fraction:
    """Parse a positive (strictly greater than zero) rational weight."""
    weight = parse_rational(value, field=f"weights[{source_id}]")
    if weight <= 0:
        raise ValueError(f"weights[{source_id}]: weight must be a positive rational, got {value!r}")
    return weight


def rational_str(value: Fraction) -> str:
    """Stable exact string form used in JSON output."""
    return str(value)


__all__ = [
    "Fraction",
    "fractions",
    "parse_rational",
    "parse_score",
    "parse_weight",
    "rational_str",
]
