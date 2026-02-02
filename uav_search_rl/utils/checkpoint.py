from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path / "checkpoint.pt")
    config = payload.get("config")
    if config is not None:
        with (path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path / "checkpoint.pt", map_location=map_location)
