"""Reproduce the U3 figures and results table.

Run with ``make u3`` or ``uv run python scripts/u3_figures.py``. Writes three
figures under ``figures/`` and ``results/u3_metrics.json``. The sweep takes
about four minutes.

Needs the MEArec recording (``make mearec``), the Nason files and the NoMAD
files.
"""

from __future__ import annotations

import json

from neurochain.analysis.u3 import compute_u3
from neurochain.analysis.u3_plots import plot_bridge, plot_pareto, plot_power
from neurochain.data import nason
from neurochain.data.mearec import load_mearec
from neurochain.paths import figures_dir, results_dir


def main() -> None:
    """Run the sweep, then write the figures and metrics."""
    in_vivo = nason.load_in_vivo()
    summary = compute_u3(load_mearec(), broadband=(in_vivo.broadband_uv, in_vivo.fs_hz))
    metrics_path = results_dir() / "u3_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    for plot, name in (
        (plot_pareto, "u3_pareto.png"),
        (plot_power, "u3_power_breakdown.png"),
        (plot_bridge, "u3_bridge.png"),
    ):
        print(f"wrote {plot(summary, figures_dir() / name)}")


if __name__ == "__main__":
    main()
