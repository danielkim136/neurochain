"""Reproduce the U6 figures and results table.

Run with ``make u6`` or ``uv run python scripts/u6_figures.py``. Writes five
figures under ``figures/`` and ``results/u6_metrics.json``. Takes about four
minutes; ``--plots-only`` redraws the figures from the metrics file in seconds.

Needs the MEArec recording (``make mearec``), the Nason files, both NoMAD
sessions and ``results/u5_metrics.json`` (committed).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from neurochain.analysis import u3
from neurochain.analysis.u6 import compute_u6
from neurochain.analysis.u6_plots import (
    plot_cross_session,
    plot_model,
    plot_robustness,
    plot_stimulation,
    plot_weeks,
)
from neurochain.data import nason, nomad
from neurochain.data.mearec import load_mearec
from neurochain.paths import figures_dir, results_dir


def load_inputs() -> dict[str, Any]:
    """Load every dataset U6 uses, as keyword arguments of ``compute_u6``."""
    sessions = sorted(nomad.load_sessions(), key=lambda s: s.start_time_ns)
    early, late = (u3.nomad_set(s, u3.U3Config().nomad_rebin) for s in sessions)
    finger = u3.nason_set()
    real = nason.load_in_vivo()
    u5_summary = json.loads((results_dir() / "u5_metrics.json").read_text())
    return {
        "sets": {finger.name: finger, early.name: early},
        "session_pair": (early, late),
        "days_apart": abs(nomad.days_between(sessions[0], sessions[1])),
        "stim_data": finger,
        "u5_signal": u5_summary["signal"],
        "broadband": (real.broadband_uv, real.fs_hz),
    }


def main() -> None:
    """Run the analysis, then write the figures and metrics."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--plots-only", action="store_true", help="redraw from results/u6_metrics.json"
    )
    args = parser.parse_args()
    metrics_path = results_dir() / "u6_metrics.json"
    if args.plots_only:
        summary = json.loads(metrics_path.read_text())
    else:
        summary = compute_u6(load_mearec(), **load_inputs())
        metrics_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {metrics_path}")
        summary = json.loads(metrics_path.read_text())
    for plot, name in (
        (plot_model, "u6_aging_model.png"),
        (plot_weeks, "u6_r2_vs_weeks.png"),
        (plot_cross_session, "u6_cross_session.png"),
        (plot_robustness, "u6_robustness.png"),
        (plot_stimulation, "u6_stimulation.png"),
    ):
        print(f"wrote {plot(summary, figures_dir() / name)}")


if __name__ == "__main__":
    main()
