from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "roi_config.yaml"


class ROIEntry(TypedDict):
    num_classes: int
    input_size: tuple[int, int]
    classes: list[str]


def _load() -> dict[str, ROIEntry]:
    with _CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f)
    return {
        roi: ROIEntry(
            num_classes=cfg["num_classes"],
            input_size=tuple(cfg["input_size"]),
            classes=cfg["classes"],
        )
        for roi, cfg in raw["rois"].items()
    }


ROI_CONFIG: dict[str, ROIEntry] = _load()
