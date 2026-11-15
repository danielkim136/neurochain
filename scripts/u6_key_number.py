"""Print the U6 key numbers, optionally with one setting changed.

With no options this prints, from ``results/u6_metrics.json``, what the aging
model does to the electrode and the features, decoding R² at week 0, at the
key week and at the last week for each chain, the real NoMAD session pair, the
stimulation test and the stabiliser's cost. With an option it reruns the
analysis with that setting changed (at three weeks, without refits and without
the robustness table), which is how the prediction checks in
``docs/U6_design.md`` are answered::

    uv run python scripts/u6_key_number.py --shift 6
    uv run python scripts/u6_key_number.py --latents 20
    uv run python scripts/u6_key_number.py --warm-start
    uv run python scripts/u6_key_number.py --decline 0.048
    uv run python scripts/u6_key_number.py --impedance-factor 8

A changed run takes about two minutes. Run one at a time: each loads the
0.9 GB MEArec recording and starts eight workers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from neurochain.analysis.u6 import U6Config, compute_u6
from neurochain.data.mearec import load_mearec
from neurochain.paths import results_dir

sys.path.insert(0, str(Path(__file__).parent))
from u6_figures import load_inputs

CHAIN_NAMES = {"sbp": "0.48 µW chain (SBP)", "crossings": "conventional chain (crossings)"}
MODE_NAMES = {
    "fixed": "left alone",
    "mean": "running mean",
    "zscore": "running z-score",
    "refit": "refitted",
    "batch": "whole-session z-score",
    "latent": "latents, not aligned",
    "aligned": "latents, aligned",
    "within": "refitted (within session)",
    "gated mean": "gated running mean",
    "gated zscore": "gated running z-score",
    "no stimulation": "no stimulation",
}


def _interval(row: dict[str, Any], key: str) -> str:
    return f"{row[key]:+.3f} [{row[key + '_low']:+.3f}, {row[key + '_high']:+.3f}]"


def show(title: str, m: dict[str, Any]) -> None:
    """Print the headline numbers of one summary."""
    print(title)
    key, model = m["key"], m["model"]
    first, last = model["states"][0], model["states"][-1]
    at = min(
        range(len(model["fine_weeks"])), key=lambda i: abs(model["fine_weeks"][i] - key["week"])
    )
    print(
        f"  electrode: |Z| at 1 kHz {first['z_1khz_kohm']:.0f} -> {last['z_1khz_kohm']:.0f} kΩ "
        f"(Rs {first['rs_kohm']:.1f} -> {last['rs_kohm']:.0f} kΩ); conventional chain's front-end "
        f"noise {first['frontend_noise_uv']:.2f} -> {last['frontend_noise_uv']:.2f} µV "
        f"(300 Hz to 7.5 kHz, input-referred); SBP noise power ratio "
        f"{first['sbp_noise_ratio']:.3f} -> {last['sbp_noise_ratio']:.3f}"
    )
    print(
        f"  spike amplitude at week {key['week']:g}: median {model['amplitude']['p50'][at]:.2f}, "
        f"10th to 90th percentile {model['amplitude']['p10'][at]:.2f} to "
        f"{model['amplitude']['p90'][at]:.2f} of week 0"
    )
    settled = m["response"]["states"][m["response"]["settled"]]
    grid = m["response"]["amplitude_grid"]
    print(
        f"  MEArec, settled electrode (|Z| x{settled['impedance_factor']:g}): crossings kept "
        + ", ".join(
            f"{settled['p_keep'][grid.index(a)]:.0%} at amplitude {a:g}"
            for a in m["config"]["check_amplitudes"]
        )
    )
    for amplitude, entry in m["bridge_check"]["amplitudes"].items():
        print(
            f"  bridge check at amplitude {amplitude}: correlation with true spike counts, "
            f"SBP {entry['sbp_actual']['r']:.3f} simulated / {entry['sbp_predicted']['r']:.3f} "
            f"bridged; crossings {entry['crossings_actual']['r']:.3f} / "
            f"{entry['crossings_predicted']['r']:.3f}"
        )
    rows = m["decoding"]["rows"]
    for condition, info in m["decoding"]["conditions"].items():
        print(
            f"  {info['dataset']}, {CHAIN_NAMES[info['chain']]}, {info['chain_uw']:.2f} µW "
            f"per channel:"
        )
        for decoder, curve in key["curves"][condition].items():
            print(
                f"    {decoder:7s} week 0: R² {curve['week0']:.3f} "
                f"({curve['week0_zscore']:.3f} behind the running z-score)"
            )
            for week in (key["week"], key["last_week"]):
                chosen = [
                    r
                    for r in rows
                    if (r["variation"], r["condition"], r["decoder"], r["week"])
                    == ("baseline", condition, decoder, week)
                ]
                fixed = next(r for r in chosen if r["mode"] == "fixed")
                print(
                    f"      week {week:3g}: left alone {fixed['r2']:.3f} "
                    f"(change {_interval(fixed, 'loss')})"
                )
                for row in chosen:
                    if row["mode"] != "fixed":
                        print(
                            f"                {MODE_NAMES[row['mode']]:16s} {row['r2']:.3f} "
                            f"(against left alone {_interval(row, 'gain')})"
                        )
    cross = m.get("cross_session")
    if cross:
        print(
            f"  real pair: fitted on {cross['train']}, tested on {cross['test']}, "
            f"{cross['days_apart']:.1f} days later ({cross['n_latents']} latents, axes from the "
            f"first {cross['calibration_s']:.0f} s):"
        )
        for row in cross["rows"]:
            print(
                f"    {row['decoder']:7s} {MODE_NAMES[row['mode']]:26s} R² {row['r2']:6.3f} "
                f"(against left alone {_interval(row, 'gain')})"
            )
        shift = cross["shift"]
        print(
            f"    channel firing rates, later over earlier: median "
            f"{shift['rate_ratio_median']:.2f}, 10th to 90th percentile "
            f"{shift['rate_ratio_p10']:.2f} to {shift['rate_ratio_p90']:.2f}; "
            f"{shift['share_rate_up']:.0%} of channels rose"
        )
    stim = m.get("stimulation")
    if stim:
        print(f"  stimulation ({stim['scenario']}, {stim['treatment']} canceller, 0.48 µW chain):")
        for row in stim["rows"]:
            extra = f" (against no stimulation {_interval(row, 'loss')})" if "loss" in row else ""
            print(
                f"    {row['decoder']:7s} {MODE_NAMES[row['mode']]:22s} R² {row['r2']:.3f}{extra}"
            )
    cost = m["cost"]
    print(
        f"  stabiliser: {cost['modes']['zscore']['uw_per_channel']:.5f} µW per channel "
        f"({cost['chains']['sbp']['zscore_share_of_chain']:.2%} of the "
        f"{cost['chains']['sbp']['chain_uw']:.2f} µW chain), "
        f"{cost['modes']['zscore']['register_bits_per_channel']} register bits, no multiplier; "
        f"latent projection {cost['modes']['aligned']['uw_per_channel']:.5f} µW per channel"
    )


def main() -> None:
    """Print the committed result, and a changed one if any option is given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--shift", type=int, help="stabiliser time constant, 2**shift bins (10)")
    parser.add_argument("--max-gain", type=float, help="limit on the running scale's gain (4)")
    parser.add_argument("--warm-start", action="store_true", help="start from the aged statistics")
    parser.add_argument("--latents", type=int, help="latent dimensions in the alignment (10)")
    parser.add_argument(
        "--calibration-s", type=float, help="unlabelled seconds for alignment (300)"
    )
    parser.add_argument("--decline", type=float, help="amplitude lost per month (0.024)")
    parser.add_argument("--spread", type=float, help="spread of decline rates across channels (1)")
    parser.add_argument("--impedance-factor", type=float, help="settled |Z| ratio at 1 kHz (2.48)")
    parser.add_argument("--interface", action="store_true", help="put the rise in the interface")
    parser.add_argument("--background-factor", type=float, help="settled background ratio (1)")
    parser.add_argument("--fixed-threshold", action="store_true", help="keep week-0 thresholds")
    args = parser.parse_args()

    committed = json.loads((results_dir() / "u6_metrics.json").read_text())
    show("U6 key numbers (committed)", committed)

    config = U6Config()
    aging: dict[str, Any] = {}
    changes = []
    if args.shift is not None:
        config = replace(config, shift=args.shift)
        changes.append(f"shift = {args.shift}")
    if args.max_gain is not None:
        config = replace(config, max_gain=args.max_gain)
        changes.append(f"gain limit = {args.max_gain:g}")
    if args.warm_start:
        config = replace(config, warm_start=True)
        changes.append("warm start")
    if args.latents is not None:
        config = replace(config, n_latents=args.latents)
        changes.append(f"latents = {args.latents}")
    if args.calibration_s is not None:
        config = replace(config, align_calibration_s=args.calibration_s)
        changes.append(f"alignment calibration = {args.calibration_s:g} s")
    if args.decline is not None:
        aging["amplitude_decline_per_month"] = args.decline
        changes.append(f"amplitude decline = {args.decline:g} per month")
    if args.spread is not None:
        aging["decline_spread"] = args.spread
        changes.append(f"decline spread = {args.spread:g}")
    if args.impedance_factor is not None:
        aging["impedance_factor_final"] = args.impedance_factor
        changes.append(f"impedance factor = {args.impedance_factor:g}")
    if args.interface:
        aging["encapsulation"] = "interface"
        changes.append("rise in the interface")
    if args.background_factor is not None:
        aging["background_factor_final"] = args.background_factor
        changes.append(f"background factor = {args.background_factor:g}")
    if args.fixed_threshold:
        config = replace(config, rethreshold=False)
        changes.append("thresholds fixed at week 0")
    if not changes:
        return

    config = replace(
        config,
        aging=replace(config.aging, **aging),
        weeks=(0.0, config.key_week, max(config.weeks)),
        refit=False,
    )
    changed = compute_u6(load_mearec(), config=config, with_variations=False, **load_inputs())
    print()
    show(f"Changed: {', '.join(changes)}", changed)


if __name__ == "__main__":
    main()
