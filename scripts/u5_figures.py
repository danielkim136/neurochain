"""Reproduce the U5 figures, results table and reference netlist.

Run with ``make u5`` or ``uv run python scripts/u5_figures.py``. Writes four
figures under ``figures/``, ``results/u5_metrics.json`` and
``spice/u5_saturation.cir``. Takes about two minutes.

Needs the MEArec recording (``make mearec``), the Nason files,
``results/u3_metrics.json`` (committed) and, for the SPICE comparison, ngspice
(``brew install ngspice``). ``--strict`` makes a missing ngspice an error
instead of a gap in the recovery table.
"""

from __future__ import annotations

import argparse
import json

from neurochain.analysis import u3
from neurochain.analysis.u5 import compute_u5, reference_netlist
from neurochain.analysis.u5_plots import (
    plot_artifact,
    plot_cancellation,
    plot_decoding,
    plot_recovery,
)
from neurochain.data import nason
from neurochain.data.mearec import load_mearec
from neurochain.frontend import spice
from neurochain.paths import figures_dir, repo_root, results_dir


def main() -> None:
    """Run the analysis, then write the figures, metrics and netlist."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--strict", action="store_true", help="fail if ngspice is missing")
    args = parser.parse_args()
    have_spice = spice.ngspice_available()
    if args.strict and not have_spice:
        raise SystemExit("ngspice is not installed (macOS: `brew install ngspice`)")
    u3_summary = json.loads((results_dir() / "u3_metrics.json").read_text())
    real = nason.load_in_vivo()
    summary = compute_u5(
        load_mearec(),
        u3_summary=u3_summary,
        data=u3.nason_set(),
        broadband=(real.broadband_uv, real.fs_hz),
        with_spice=have_spice,
    )
    metrics_path = results_dir() / "u5_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    netlist_path = repo_root() / "spice" / "u5_saturation.cir"
    netlist_path.write_text(reference_netlist())
    print(f"wrote {netlist_path}")
    for plot, name in (
        (plot_artifact, "u5_artifact.png"),
        (plot_recovery, "u5_recovery.png"),
        (plot_cancellation, "u5_cancellation.png"),
        (plot_decoding, "u5_decoding.png"),
    ):
        print(f"wrote {plot(summary, figures_dir() / name)}")


if __name__ == "__main__":
    main()
