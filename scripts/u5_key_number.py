"""Print the U5 key numbers, optionally with one setting changed.

With no options this prints, from ``results/u5_metrics.json``, how large the
artifact is, how long each chain takes to recover, what each treatment leaves
in the features, and what that does to decoding. With an option it reruns the
signal-level measurement on MEArec and the decoding on the stimulating array,
which is how the prediction checks in ``docs/U5_design.md`` are answered::

    uv run python scripts/u5_key_number.py --adapt-shift 1
    uv run python scripts/u5_key_number.py --jitter-us 100
    uv run python scripts/u5_key_number.py --rate-hz 300
    uv run python scripts/u5_key_number.py --imbalance 0.05

A changed run takes about a minute; ``--no-decode`` stops after the signal
level.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from neurochain.analysis import u3
from neurochain.analysis.u5 import TREATMENTS, U5Config, compute_u5
from neurochain.data.mearec import load_mearec
from neurochain.paths import results_dir

CHAIN_NAMES = {"sbp": "0.48 µW chain (SBP)", "crossings": "conventional chain (crossings)"}


def _interval(row: dict[str, Any], key: str) -> str:
    return f"{row[key]:+.3f} [{row[key + '_low']:+.3f}, {row[key + '_high']:+.3f}]"


def show(title: str, m: dict[str, Any]) -> None:
    """Print the headline numbers of one summary."""
    print(title)
    key, sig, rec = m["key"], m["signal"], m["recovery"]["chains"]
    index = sig["amplitudes_mv"].index(key["example_mv"])
    for s in m["propagation"]["scenarios"]:
        print(
            f"  artifact, {s['name']:11s}: median {s['median_mv']:.2f} mV "
            f"({s['min_mv']:.2f} to {s['max_mv']:.2f}), {s['share_above_adc']:.0%} of channels "
            f"past the ADC range, {s['share_above_amplifier']:.0%} past the amplifier's"
        )
    for name, chain in rec.items():
        spice = chain["spice_ms"][index]
        spice_text = "not run" if spice is None else f"{spice:.2f} ms"
        print(
            f"  recovery at {key['example_mv']:g} mV, {CHAIN_NAMES[name]}: "
            f"{chain['model_ms'][index]:.2f} ms (ngspice {spice_text}); "
            f"ADC back in range after {chain['in_range_ms'][index]:.2f} ms"
        )
    print(
        f"  {sig['n_pulses']} pulses at {sig['realised_rate_hz']:.0f} per second on "
        f"{sig['n_channels']} channels; at {key['example_mv']:g} mV:"
    )
    for treatment in TREATMENTS:
        sbp, crossings = sig["sbp"][treatment], sig["crossings"][treatment]
        print(
            f"    {treatment:9s} SBP shift {sbp['bias'][index]:+7.3f} floors, spread "
            f"{sbp['sd'][index]:.3f}, samples kept {sbp['valid'][index]:.0%}, residual "
            f"{sbp['residual_uv'][index]:6.1f} µV | crossings kept "
            f"{crossings['p_keep'][index]:.0%}, false positives "
            f"+{crossings['extra_rate_hz'][index]:.1f} per second"
        )
    decode = m.get("decoding")
    if decode:
        scenario, decoder = key["scenario"], key["decoder"]
        print(f"  decoding on stimulated bins, {scenario}, {decoder}:")
        for chain in CHAIN_NAMES:
            print(
                f"    {CHAIN_NAMES[chain]}, no stimulation: "
                f"R² {decode['no_stimulation_r2'][chain][decoder]:.3f}"
            )
            for row in decode["rows"]:
                if (row["chain"], row["scenario"], row["decoder"]) == (chain, scenario, decoder):
                    print(
                        f"      {row['treatment']:9s} fitted {row['fitted']:19s} "
                        f"R² {row['r2']:7.3f}  change {_interval(row, 'loss')}"
                    )
    cost = m["cost"]["chains"]["sbp"]
    print(
        f"  canceller: {cost['canceller_uw']:.4f} µW per channel "
        f"({cost['share_of_chain']:.1%} of the {cost['chain_uw']:.2f} µW chain), "
        f"{cost['template_bits_per_channel']} bits of template, no multiplier"
    )


def main() -> None:
    """Print the committed result, and a changed one if any option is given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--adapt-shift", type=int, help="adaptation shift of the template (4)")
    parser.add_argument("--jitter-us", type=float, help="pulse timing jitter, ±µs (0)")
    parser.add_argument("--rate-hz", type=float, help="pulse rate inside a train (100)")
    parser.add_argument("--imbalance", type=float, help="charge imbalance of the pulse (0.01)")
    parser.add_argument("--current-ua", type=float, help="pulse amplitude, µA (60)")
    parser.add_argument("--no-decode", action="store_true", help="skip the decoding step")
    args = parser.parse_args()

    committed = json.loads((results_dir() / "u5_metrics.json").read_text())
    show("U5 key numbers (committed)", committed)

    config = U5Config()
    changes = []
    if args.adapt_shift is not None:
        config = replace(config, adapt_shift=args.adapt_shift)
        changes.append(f"adaptation shift = {args.adapt_shift}")
    if args.jitter_us is not None:
        config = replace(config, jitter_s=args.jitter_us * 1e-6)
        changes.append(f"jitter = ±{args.jitter_us:g} µs")
    if args.rate_hz is not None:
        config = replace(config, rate_hz=args.rate_hz)
        changes.append(f"rate = {args.rate_hz:g} Hz")
    if args.imbalance is not None:
        config = replace(config, pulse=replace(config.pulse, imbalance=args.imbalance))
        changes.append(f"imbalance = {args.imbalance:g}")
    if args.current_ua is not None:
        config = replace(config, pulse=replace(config.pulse, amplitude_ua=args.current_ua))
        changes.append(f"current = {args.current_ua:g} µA")
    if not changes:
        return

    config = replace(config, scenarios_mm=config.scenarios_mm[:1])
    decode = not args.no_decode
    changed = compute_u5(
        load_mearec(),
        config=config,
        u3_summary=json.loads((results_dir() / "u3_metrics.json").read_text()) if decode else None,
        data=u3.nason_set() if decode else None,
        with_spice=False,
    )
    print()
    show(f"Changed: {', '.join(changes)}", changed)


if __name__ == "__main__":
    main()
