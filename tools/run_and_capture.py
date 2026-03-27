from __future__ import annotations

import argparse
import json
import os
import runpy
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Python script and save all matplotlib figures.")
    parser.add_argument("--script", required=True, help="Path to the Python script to execute.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures are written.")
    parser.add_argument("--prefix", default=None, help="Figure filename prefix (default: script stem).")
    parser.add_argument("--max-figures", type=int, default=40, help="Maximum number of figures to save.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or script_path.stem
    manifest_path = output_dir / f"{prefix}_manifest.json"

    saved_files: list[str] = []
    fig_counter = 0

    def save_open_figures(*_a, **_kw) -> None:
        nonlocal fig_counter
        for fig_num in plt.get_fignums():
            if fig_counter >= args.max_figures:
                plt.close("all")
                return
            fig_counter += 1
            fig = plt.figure(fig_num)
            filename = f"{prefix}_fig_{fig_counter:03d}.png"
            out_path = output_dir / filename
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            saved_files.append(out_path.name)
        plt.close("all")

    plt.show = save_open_figures

    old_cwd = Path.cwd()
    try:
        os.chdir(script_path.parent)
        runpy.run_path(str(script_path), run_name="__main__")
        save_open_figures()
    except Exception:
        save_open_figures()
        manifest_path.write_text(json.dumps(saved_files, indent=2), encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        os.chdir(old_cwd)

    manifest_path.write_text(json.dumps(saved_files, indent=2), encoding="utf-8")
    print(f"Saved {len(saved_files)} figure(s) for {script_path.name} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
