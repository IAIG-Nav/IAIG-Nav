"""Shared config helpers for consistent defaults."""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg: Dict[str, Any], section: str, key: str, fallback: Any) -> Any:
    return cfg.get(section, {}).get(key, fallback)
