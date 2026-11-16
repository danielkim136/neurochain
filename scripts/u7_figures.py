"""Reproduce the U7 results: bit-exactness, synthesis, power, figures.

Run with ``make u7`` or ``uv run python scripts/u7_figures.py``. Simulates the
SystemVerilog under ``rtl/`` on ten minutes of the Neuralink clips and compares
every output with the integer Python reference, synthesises the design for the
ECP5 at several channel counts, measures switching activity on a gate-level
netlist, and writes three figures under ``figures/`` and
``results/u7_metrics.json``. Takes about two minutes; ``--plots-only`` redraws
the figures from the metrics file in seconds.

Needs the Neuralink clips (``make data``), Verilator, Yosys and nextpnr-ecp5
(``docs/U7_design.md``, section 2, says how they were installed). Tool outputs
go to ``build/u7``, which git ignores.
"""

from __future__ import annotations

import argparse
import json

from neurochain.analysis.u7 import compute_u7
from neurochain.analysis.u7_plots import plot_bit_exact, plot_power, plot_synthesis
from neurochain.paths import figures_dir, results_dir


def main() -> None:
    """Run the analysis, then write the figures and metrics."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--plots-only", action="store_true", help="redraw from results/u7_metrics.json"
    )
    args = parser.parse_args()
    metrics_path = results_dir() / "u7_metrics.json"
    if not args.plots_only:
        summary = compute_u7()
        metrics_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {metrics_path}")
    summary = json.loads(metrics_path.read_text())
    for plot, name in (
        (plot_bit_exact, "u7_bit_exact.png"),
        (plot_synthesis, "u7_synthesis.png"),
        (plot_power, "u7_power.png"),
    ):
        print(f"wrote {plot(summary, figures_dir() / name)}")


if __name__ == "__main__":
    main()
