"""Project I/O helpers for figures, CSV files, and manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_ROOT = PROJECT_ROOT / "figures"
RESULTS_ROOT = PROJECT_ROOT / "results"
METRICS_ROOT = RESULTS_ROOT / "metrics"
MANIFESTS_ROOT = RESULTS_ROOT / "manifests"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(*parts: str) -> Path:
    return ensure_directory(FIGURES_ROOT / parts[0]).joinpath(*parts[1:]) if len(parts) > 1 else FIGURES_ROOT


def metrics_path(filename: str) -> Path:
    ensure_directory(METRICS_ROOT)
    return METRICS_ROOT / filename


def manifest_path(filename: str) -> Path:
    ensure_directory(MANIFESTS_ROOT)
    return MANIFESTS_ROOT / filename


def write_rows_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    ensure_directory(path.parent)
    if not rows:
        raise ValueError(f"No rows provided for {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(name: str, files: list[Path]) -> None:
    relative_files = [str(path.relative_to(PROJECT_ROOT)) for path in files]
    path = manifest_path(f"{name}_manifest.json")
    path.write_text(json.dumps(relative_files, indent=2), encoding="utf-8")
