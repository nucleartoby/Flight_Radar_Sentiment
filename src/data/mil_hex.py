import json
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Tuple

# tracked reference data the classifier cannot run without
RANGES_PATH = Path(__file__).resolve().parents[2] / "config" / "mil_ranges.json"

_starts: List[int] = []
_ranges: List[Tuple[int, int, str]] = []


def _load() -> None:
    global _starts, _ranges
    if _ranges:
        return

    with open(RANGES_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    labels = data.get("labels", {})
    parsed = sorted(
        (int(lo, 16), int(hi, 16), labels.get(lo, "unknown mil"))
        for lo, hi in data["ranges"])
    
    _ranges = parsed
    _starts = [lo for lo, _, _ in parsed]


def lookup(icao24: str) -> Optional[str]:
    if not icao24:
        return None
    try:
        addr = int(icao24, 16)
    except (ValueError, TypeError):
        return None

    _load()
    i = bisect_right(_starts, addr) - 1
    if i < 0:
        return None
    lo, hi, label = _ranges[i]
    return label if lo <= addr <= hi else None


def is_military_hex(icao24: str) -> bool:
    return lookup(icao24) is not None


def all_ranges() -> List[Tuple[int, int, str]]:
    _load()
    return list(_ranges)
