"""Print the U7 key numbers, optionally with one setting changed.

With no options this prints, from ``results/u7_metrics.json``, the bit-exactness
counts, every node's word width against what the data used, the ECP5 resources
and clock rate, and the power estimate with the chain's budget. With an option
it rebuilds and reruns with that setting changed, which is how the prediction
checks in ``docs/U7_design.md`` are answered::

    uv run python scripts/u7_key_number.py --acc-bits 18 --seconds 600
    uv run python scripts/u7_key_number.py --channels 1024
    uv run python scripts/u7_key_number.py --star
    uv run python scripts/u7_key_number.py --silent

A changed run uses 60 s of data per channel unless ``--seconds`` says otherwise,
synthesises only its own channel count, and takes one to three minutes. Run one
at a time. Needs the Neuralink clips, Verilator, Yosys and nextpnr-ecp5.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from neurochain.analysis.u7 import U7Config, compute_u7
from neurochain.paths import results_dir


def _resources(row: dict[str, Any]) -> str:
    if "error" in row:
        return f"failed: {row['error']}"
    fmax = row["fmax_mhz"]
    return (
        f"{row['lut4']} LUT4 ({row['logic_cells']} logic cells of "
        f"{row['logic_cells_available']}), {row['flip_flops']} flip-flops, {row['block_rams']} "
        f"block RAMs of {row['block_rams_available']}, {row['multipliers']} hard multipliers; "
        f"fmax {row['fmax_median_mhz']:.1f} MHz (seeds: {min(fmax):.1f} to {max(fmax):.1f})"
    )


def show(title: str, m: dict[str, Any]) -> None:
    """Print the headline numbers of one summary."""
    print(title)
    stim, real, worst, gate = m["stimulus"], m["bit_exact"], m["worst_case"], m["gate_level"]
    print(
        f"  data: {stim['source']}, {real['n_channels']} channels x {real['seconds']:.0f} s at "
        f"2 kS/s, 8 bits; {stim['n_pulses']} pulses in {stim['n_trains']} trains "
        f"({stim['stimulated_fraction']:.0%} of the time); {stim['railed_fraction']:.2%} of "
        f"samples on a rail; quiet RMS {stim['quiet_rms_lsb']:.2f} LSB"
    )
    counts = ", ".join(f"{name} {n}" for name, n in real["mismatches"].items())
    print(
        f"  RTL against Python: {real['total_mismatches']} mismatches in "
        f"{real['values_compared']:,} values ({counts}); channel tags wrong: "
        f"{real['tag_errors']}; {real['blanked_fraction']:.2%} of samples blanked"
    )
    if real["first_mismatch"]:
        first = real["first_mismatch"]
        print(
            f"    first mismatch: channel {first['channel']}, sample {first['sample']}; "
            f"{real['channels_with_mismatches']} channels affected"
        )
    print(
        f"  worst-case stimulus: {worst['total_mismatches']} mismatches in "
        f"{worst['values_compared']:,} values; gate-level netlist against RTL: "
        f"{gate['mismatches_against_rtl']} in {gate['values_compared']:,} "
        f"({gate['seconds']:.0f} s{', silent input' if gate['silent'] else ''})"
    )
    print(
        f"  simulation: {real['clock_cycles']:,} clock cycles in {real['simulation_s']:.1f} s; "
        f"Python reference {real['reference_s']:.0f} s"
    )
    print("  node                 bound   bits   peak, real data (bits)   peak, worst case")
    for row in m["widths"]:
        print(
            f"  {row['node']:<19}{row['bound']:>7}{row['bits']:>7}"
            f"{row['peak_real']:>12} ({row['bits_used_real']:>2})"
            f"{row['peak_worst_case']:>21}"
        )
    bits = m["state_bits"]
    print(
        f"  state per channel: {bits['total']} bits (template {bits['template']}, filter "
        f"{bits['filter']}, bin {bits['bin']}, control {bits['control']})"
    )
    synth = m["synthesis"]
    print(f"  {synth['device']}, speed grade {synth['speed_grade']}:")
    for row in synth["sweep"]:
        print(f"    {row['n_channels']:>5} channels: {_resources(row)}")
    for name, label in (("star_lut", '"*" in LUTs'), ("star_dsp", '"*" in hard multipliers')):
        row = synth["variants"][name]
        print(f"    {row['n_channels']:>5} channels, {label}: {_resources(row)}")
    power, key = m["power"], m["key"]
    energy, measured = power["energy"], power["measured"]
    print(
        f"  activity: {measured['toggles_per_sample']:.0f} node toggles per sample over "
        f"{gate['activity']['nets']} nets (activity {gate['activity']['activity']:.2f}); "
        f"{measured['gates']} gates, {measured['flip_flops']} flip-flops; "
        f"{measured['memory_bits_per_sample']:.0f} memory bits touched per sample"
    )
    print(
        f"  energy per sample: logic {energy['logic_pj']:.2f} pJ, clock {energy['clock_pj']:.2f} "
        f"pJ, memory {energy['memory_pj']:.2f} pJ; placeholder "
        f"{power['placeholder']['operations_per_sample']} x "
        f"{power['placeholder']['pj_per_operation']:g} pJ"
    )
    print(
        f"  digital power: {energy['total_uw']:.4f} µW per channel (dynamic "
        f"{energy['dynamic_uw']:.4f}, leakage {energy['leakage_uw']:.4f}); with every assumed "
        f"constant a tenth to ten times: {key['digital_low_uw']:.4f} to "
        f"{key['digital_high_uw']:.3f}; placeholder {key['placeholder_uw']:.3f}"
    )
    budget = power["budget"]
    print(
        f"  chain: {budget['before_uw']:.3f} µW per channel with the placeholders, "
        f"{budget['after_uw']:.3f} with the synthesised estimate and a "
        f"{power['radio']['bits_per_bin_rtl']}-bit radio word "
        f"({budget['after_low_uw']:.3f} to {budget['after_high_uw']:.3f}); amplifier "
        f"{budget['lna_share']:.0%}, digital {budget['digital_share']:.1%}"
    )


def main() -> None:
    """Parse the options, rerun if any setting changed, and print."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--acc-bits", type=int, help="filter accumulator width (20)")
    parser.add_argument("--channels", type=int, help="channels sharing the datapath (96)")
    parser.add_argument("--adapt-shift", type=int, help="template adaptation shift (4)")
    parser.add_argument("--star", action="store_true", help='write the products as "*"')
    parser.add_argument("--silent", action="store_true", help="measure activity on zero input")
    parser.add_argument("--seconds", type=float, help="seconds of data per channel")
    args = parser.parse_args()
    changed = [k for k, v in vars(args).items() if v not in (None, False) and k != "seconds"]
    if not changed and args.seconds is None:
        path = results_dir() / "u7_metrics.json"
        if not path.exists():
            raise SystemExit("results/u7_metrics.json is missing; run `make u7` first")
        show("U7 key numbers (results/u7_metrics.json)", json.loads(path.read_text()))
        return
    config = U7Config()
    spec = config.spec
    if args.acc_bits is not None:
        spec = replace(spec, y_bits_override=args.acc_bits - spec.coef_frac_bits)
    if args.channels is not None:
        spec = replace(spec, n_channels=args.channels)
    if args.adapt_shift is not None:
        spec = replace(spec, adapt_shift=args.adapt_shift)
    if args.star:
        spec = replace(spec, shift_add=False)
    config = replace(
        config,
        spec=spec,
        silent=args.silent,
        seconds=args.seconds if args.seconds is not None else 60.0,
        activity_s=max(2.0, config.activity_s * config.spec.n_channels / spec.n_channels),
    )
    described = ", ".join(changed) if changed else "nothing"
    show(
        f"U7 with {described} changed, {config.seconds:.0f} s per channel",
        compute_u7(config, sweep=False),
    )


if __name__ == "__main__":
    main()
