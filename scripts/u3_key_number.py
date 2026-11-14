"""Print the U3 key numbers, optionally under different technology constants.

With no options this prints the cheapest configuration that matches the
conventional chain on each dataset, from ``results/u3_metrics.json``. The
options change the power model's constants and recompute, without rerunning the
sweep, which is how the prediction checks in ``docs/U3_design.md`` are
answered::

    uv run python scripts/u3_key_number.py --adc-fom-fj 500
    uv run python scripts/u3_key_number.py --pj-per-bit 1000
    uv run python scripts/u3_key_number.py --nef 2
    uv run python scripts/u3_key_number.py --tolerance 0.10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from neurochain.analysis.u3 import rescore
from neurochain.paths import results_dir
from neurochain.power import PowerModel


def show(title: str, result: dict) -> None:
    """Print one rescoring as a small table."""
    print(title)
    for name, row in result.items():
        if name == "raw_stream":
            print(f"  {'raw stream':16s} {row['total_uw']:8.2f} µW per channel")
            continue
        saving = row["baseline_uw"] / row["winner_uw"]
        print(
            f"  {name:16s} conventional {row['baseline_uw']:6.2f} µW (R² {row['baseline_r2']:.3f})"
            f" -> {row['winner']}: {row['winner_uw']:.3f} µW (R² {row['winner_r2']:.3f}), "
            f"{saving:.1f}x less"
        )


def main() -> None:
    """Print the committed result, and a changed one if any option is given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adc-fom-fj", type=float, help="ADC figure of merit, fJ/step (50)")
    parser.add_argument("--pj-per-bit", type=float, help="radio energy per bit, pJ (10)")
    parser.add_argument("--pj-per-multiply", type=float, help="logic energy per multiply, pJ (1)")
    parser.add_argument("--v-supply", type=float, help="amplifier supply voltage, V (1.8)")
    parser.add_argument("--nef", type=float, help="amplifier noise efficiency factor (4)")
    parser.add_argument("--tolerance", type=float, help="accuracy a match may lose (0.05)")
    args = parser.parse_args()

    summary = json.loads((results_dir() / "u3_metrics.json").read_text())
    baseline_model = PowerModel(**summary["config"]["power"])
    baseline_tolerance = summary["config"]["tolerance"]
    show(
        "U3 key numbers: cheapest configuration within "
        f"{baseline_tolerance:.0%} of the conventional chain's R²",
        rescore(summary, baseline_model, baseline_tolerance),
    )
    changes = {k: v for k, v in vars(args).items() if v is not None}
    if not changes:
        return
    model = replace(
        baseline_model,
        adc_fom_j=(args.adc_fom_fj or baseline_model.adc_fom_j * 1e15) * 1e-15,
        energy_per_bit_j=(args.pj_per_bit or baseline_model.energy_per_bit_j * 1e12) * 1e-12,
        energy_per_multiply_j=(args.pj_per_multiply or baseline_model.energy_per_multiply_j * 1e12)
        * 1e-12,
        v_supply_v=args.v_supply or baseline_model.v_supply_v,
        nef=args.nef or baseline_model.nef,
    )
    tolerance = args.tolerance if args.tolerance is not None else baseline_tolerance
    print()
    show(
        "Changed: " + ", ".join(f"{k} = {v:g}" for k, v in changes.items()),
        rescore(summary, model, tolerance),
    )


if __name__ == "__main__":
    main()
