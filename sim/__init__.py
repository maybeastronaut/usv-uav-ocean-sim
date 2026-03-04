from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_src_sim = Path(__file__).resolve().parent.parent / "src" / "sim"
if _src_sim.is_dir():
    __path__.append(str(_src_sim))
