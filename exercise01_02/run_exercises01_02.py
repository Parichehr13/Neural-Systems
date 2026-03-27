from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "exercises01_02.py"
FIGURES_DIR = BASE_DIR / "figures"
MANIFEST_PATH = BASE_DIR / "figures_manifest.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

figure_counter = 0
saved_files: list[str] = []


def save_open_figures(*_args, **_kwargs) -> None:
    global figure_counter

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        figure_counter += 1
        filename = f"fig_{figure_counter:03d}.png"
        output_path = FIGURES_DIR / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_files.append(str(output_path.relative_to(BASE_DIR)))
    plt.close("all")


plt.show = save_open_figures

runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
save_open_figures()

MANIFEST_PATH.write_text(json.dumps(saved_files, indent=2), encoding="utf-8")

print(f"Saved {len(saved_files)} figures to: {FIGURES_DIR}")
