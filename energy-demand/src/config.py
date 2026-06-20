"""Project paths. Import these everywhere instead of hard-coding relative paths,
so notebooks (in notebooks/) and scripts (anywhere) resolve data the same way."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

for _p in (DATA_RAW, DATA_PROC, FIGS):
    _p.mkdir(parents=True, exist_ok=True)
