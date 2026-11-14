"""Reproduce the U4 figures and results table.

Run with ``make u4`` or ``uv run python scripts/u4_figures.py``. Writes three
figures under ``figures/`` and ``results/u4_metrics.json``. Takes about three
minutes, most of it training the GRU.

Needs ``results/u3_metrics.json`` (committed; rebuilt by ``make u3``), the Nason
files and the NoMAD files.
"""

from __future__ import annotations

import json

from neurochain.analysis.u4 import compute_u4
from neurochain.analysis.u4_plots import plot_cross_session, plot_decoders, plot_traces
from neurochain.paths import figures_dir, results_dir


def main() -> None:
    """Score the decoders, then write the figures and metrics."""
    u3_summary = json.loads((results_dir() / "u3_metrics.json").read_text())
    summary = compute_u4(u3_summary)
    metrics_path = results_dir() / "u4_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    for plot, name in (
        (plot_decoders, "u4_decoders.png"),
        (plot_traces, "u4_traces.png"),
        (plot_cross_session, "u4_cross_session.png"),
    ):
        print(f"wrote {plot(summary, figures_dir() / name)}")


if __name__ == "__main__":
    main()
