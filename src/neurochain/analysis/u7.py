"""U7 analysis: the SBP chain in SystemVerilog, proven against Python and costed.

Four steps:

1. **Bit-exactness.** Ten minutes of real recordings (the Neuralink clips,
   turned into 8-bit codes at 2 kS/s with stimulation artifacts added) go
   through the Verilator model of ``rtl/nc_sbp_chain.sv`` and through the
   integer Python chain (:meth:`neurochain.rtl.spec.ChainSpec.reference`).
   Every canceller output, blanking flag, filter output, bin sum and bin count
   is compared; the criterion is zero mismatches. A second, directed stimulus
   drives the registers to their worst-case values.
2. **Synthesis.** Yosys and nextpnr build the design for a Lattice ECP5
   LFE5U-25F at several channel counts: LUTs, flip-flops, block RAM and the
   clock rate the slowest path allows.
3. **Activity.** Yosys also writes a NAND/NOR netlist; Verilator simulates it
   on the same codes (it must again match bit for bit) and counts how often
   every node switches.
4. **Power.** Counts and activity times stated energies give µW per channel
   (:mod:`neurochain.rtl.energy`), which replaces the 1 pJ-per-multiply
   placeholder of U3's power model in the 0.48 µW budget.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from neurochain.power import ChainConfig, PowerModel, chain_power
from neurochain.rtl import tools
from neurochain.rtl.energy import EnergyModel, chain_energy, summarise_activity
from neurochain.rtl.spec import ChainSpec
from neurochain.rtl.stimulus import Stimulus, neuralink_stimulus, worst_case_stimulus

SIGNALS = ("cancelled", "valid", "filtered", "sum", "count")
HOROWITZ_ADD_J = 0.1e-12
"""Energy of a 32-bit addition at 45 nm and 0.9 V (Horowitz, 2014)."""
CANCELLER_MODEL_ADDS = 2
"""Additions U5 charged the canceller, each as a full multiply-accumulate."""
RADIO_WORD_BITS = 12
"""Bits per SBP value U3's power model sends (:class:`neurochain.power.ChainConfig`)."""


@dataclass(frozen=True)
class U7Config:
    """Parameters of the U7 analysis.

    Attributes:
        seed: Seed for the clip order, the pulse trains and the artifact sizes.
        spec: The chain: channel count, word lengths, filter, bin.
        seconds: Length of the real-data comparison per channel.
        activity_s: Length of the gate-level run that counts toggles.
        silent: Count toggles on all-zero codes with no pulses instead.
        channel_counts: Channel counts synthesised for the ECP5.
        pnr_seeds: Placement seeds; the clock rate is reported for each.
        freq_mhz: Clock rate nextpnr aims for.
        example_s: Length of the stretch of signals kept for the figure.
        energy: Constants of the digital energy estimate.
        power: Constants of U3's power model.
        sbp_current_a: Supply current of the SBP chain's amplifier (U3's choice).
        max_workers: Processes running the Python reference.
    """

    seed: int = 0
    spec: ChainSpec = ChainSpec()  # noqa: RUF009 - frozen dataclass, safe as a default
    seconds: float = 600.0
    activity_s: float = 20.0
    silent: bool = False
    channel_counts: tuple[int, ...] = (1, 8, 32, 96, 192, 1024, 2048)
    pnr_seeds: tuple[int, ...] = (1, 2, 3)
    freq_mhz: float = 50.0
    example_s: float = 0.4
    energy: EnergyModel = EnergyModel()  # noqa: RUF009
    power: PowerModel = PowerModel()  # noqa: RUF009
    sbp_current_a: float = 0.25e-6
    max_workers: int = 8


# ----------------------------------------------------------------------- bit-exactness

CompareJob = tuple[ChainSpec, Any, Any, str, int]


def _compare_channel(job: CompareJob) -> dict[str, Any]:
    """Run the Python reference on one channel and compare the simulation with it."""
    spec, codes, onsets, prefix, channel = job
    out = tools.load_output(Path(prefix), spec, {})
    ref = spec.reference(codes, onsets)
    cancelled, filtered = out.cancelled(channel), out.filtered(channel)
    valid = np.asarray(out.ok[:, channel], dtype=bool)
    wrong = {
        "cancelled": cancelled != ref.cancelled[0],
        "valid": valid != ref.valid[0],
        "filtered": filtered != ref.filtered[0],
        "sum": np.asarray(out.sums[:, channel], dtype=np.int64) != ref.sums[0],
        "count": np.asarray(out.counts[:, channel], dtype=np.int64) != ref.counts[0],
    }
    per_sample = wrong["cancelled"] | wrong["valid"] | wrong["filtered"]
    wide = np.asarray(codes, dtype=np.int64)
    return {
        "channel": channel,
        "samples": int(codes.size),
        "bins": int(ref.sums.shape[1]),
        "mismatches": {name: int(flags.sum()) for name, flags in wrong.items()},
        "first_mismatch": int(np.argmax(per_sample)) if per_sample.any() else None,
        "peak": {
            # Folded as two's complement does: -128 needs no more bits than 127.
            "code": int(max(wide.max(), -wide.min() - 1)),
            "cancelled": int(np.abs(ref.cancelled).max()),
            "filtered": int(np.abs(ref.filtered).max()),
            "accumulator": int(ref.accumulator_peak),
            "sum": int(ref.sums.max()) if ref.sums.size else 0,
        },
        "blanked": int((~ref.valid).sum()),
        "sum_total": int(ref.sums.sum()),
        "sums_over_radio_word": int((ref.sums >= 1 << RADIO_WORD_BITS).sum()),
        "count_min": int(ref.counts.min()) if ref.counts.size else 0,
    }


def bit_exact(
    stimulus: Stimulus, spec: ChainSpec, directory: Path, *, name: str, max_workers: int
) -> tuple[dict[str, Any], tools.SimOutput]:
    """Simulate the RTL on a stimulus and compare every output with Python.

    Args:
        stimulus: Codes and pulse onsets.
        spec: The chain.
        directory: Build directory.
        name: Prefix of the files this run writes.
        max_workers: Processes for the reference; 1 runs it in this process.

    Returns:
        ``(summary, output)``: mismatch counts per signal, the peaks every node
        reached, and the simulation's outputs.
    """
    program = tools.build_model(spec, directory / "verilator_rtl")
    prefix = directory / name
    out = tools.run_model(program, stimulus, spec, prefix)
    n_expected = stimulus.n_samples * spec.n_channels
    if out.report["samples_out"] != n_expected:
        raise tools.ToolError(
            f"the model returned {out.report['samples_out']} samples, expected {n_expected}"
        )
    jobs: list[CompareJob] = [
        (spec, stimulus.codes[ch], stimulus.onsets, str(prefix), ch)
        for ch in range(spec.n_channels)
    ]
    start = time.perf_counter()
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(_compare_channel, jobs, chunksize=1))
    else:
        rows = [_compare_channel(job) for job in jobs]
    reference_s = time.perf_counter() - start
    mismatches = {s: sum(r["mismatches"][s] for r in rows) for s in SIGNALS}
    failing = [r for r in rows if sum(r["mismatches"].values())]
    samples = sum(r["samples"] for r in rows)
    bins = sum(r["bins"] for r in rows)
    summary = {
        "n_channels": spec.n_channels,
        "seconds": stimulus.seconds,
        "samples_compared": samples,
        "bins_compared": bins,
        "values_compared": 3 * samples + 2 * bins,
        "mismatches": mismatches,
        "total_mismatches": sum(mismatches.values()),
        "channels_with_mismatches": len(failing),
        "first_mismatch": (
            {"channel": failing[0]["channel"], "sample": failing[0]["first_mismatch"]}
            if failing
            else None
        ),
        "tag_errors": int(out.report["tag_errors"]),
        "peak": {k: max(r["peak"][k] for r in rows) for k in rows[0]["peak"]},
        "blanked_fraction": sum(r["blanked"] for r in rows) / samples,
        "mean_sum": sum(r["sum_total"] for r in rows) / max(bins, 1),
        "radio_word_bits": RADIO_WORD_BITS,
        "bins_over_radio_word": sum(r["sums_over_radio_word"] for r in rows) / max(bins, 1),
        "count_min": min(r["count_min"] for r in rows),
        "clock_cycles": int(out.report["cycles"]),
        "simulation_s": float(out.report["wall_s"]),
        "reference_s": reference_s,
    }
    return summary, out


def width_table(
    spec: ChainSpec, real: dict[str, Any], worst: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """List every node's worst-case bound, its register width and the peaks seen.

    ``bits_used`` is the width the real data's peak would have needed; the gap
    to ``bits`` is headroom that the worst case, not the data, pays for.
    """
    frac = spec.coef_frac_bits
    nodes: list[tuple[str, str, int, int, bool]] = [
        ("ADC code", "code", spec.code_max, spec.code_bits, True),
        ("canceller output", "cancelled", spec.cancel_bound, spec.cancel_bits, True),
        (
            "filter accumulator",
            "accumulator",
            (spec.y_bound << frac) + (1 << (frac - 1)),
            spec.accumulator_bits,
            True,
        ),
        ("filter output", "filtered", spec.y_bound, spec.y_bits, True),
        ("bin sum", "sum", spec.sum_bound, spec.sum_bits, False),
    ]
    rows = []
    for label, key, bound, bits, signed in nodes:
        peak = int(real["peak"][key])
        rows.append(
            {
                "node": label,
                "signed": signed,
                "bound": int(bound),
                "bits": int(bits),
                "peak_real": peak,
                "bits_used_real": peak.bit_length() + int(signed),
                "peak_worst_case": int(worst["peak"][key]) if worst else None,
            }
        )
    return rows


def example_traces(
    stimulus: Stimulus, out: tools.SimOutput, spec: ChainSpec, seconds: float
) -> dict[str, Any]:
    """Cut a short stretch of one channel around the start of a pulse train.

    The stretch starts 50 ms before the first pulse that follows a pause, at
    least a second into the record. The channel is the one whose largest code
    in the stretch is nearest 100: a visible artifact that does not sit on the
    rail throughout.
    """
    fs = spec.fs_hz
    gaps = np.flatnonzero(np.diff(stimulus.onsets) > 2 * spec.n_taps) + 1
    late = [int(stimulus.onsets[i]) for i in gaps if stimulus.onsets[i] > fs]
    onset = late[0] if late else (int(stimulus.onsets[0]) if stimulus.onsets.size else 0)
    per_bin = spec.samples_per_bin
    start = max(0, (onset - round(0.05 * fs)) // per_bin * per_bin)
    stop = min(stimulus.n_samples, start + round(seconds * fs) // per_bin * per_bin)
    peaks = np.abs(stimulus.codes[:, start:stop].astype(np.int64)).max(axis=1)
    channel = int(np.argmin(np.abs(peaks - 100)))
    codes = stimulus.codes[channel, :stop]
    ref = spec.reference(codes, stimulus.onsets[stimulus.onsets < stop])
    bins = slice(start // per_bin, stop // per_bin)
    scale = 1 << spec.out_frac_bits
    return {
        "channel": channel,
        "start_s": start / fs,
        "fs_hz": fs,
        "scale": scale,
        "samples_per_bin": per_bin,
        "onsets": [int(n) - start for n in stimulus.onsets if start <= n < stop],
        "codes": codes[start:stop].astype(int).tolist(),
        "rtl": {
            "cancelled": out.cancelled(channel)[start:stop].tolist(),
            "valid": np.asarray(out.ok[start:stop, channel]).astype(int).tolist(),
            "filtered": out.filtered(channel)[start:stop].tolist(),
            "sum": np.asarray(out.sums[bins, channel]).astype(int).tolist(),
            "count": np.asarray(out.counts[bins, channel]).astype(int).tolist(),
        },
        "python": {
            "cancelled": ref.cancelled[0, start:stop].tolist(),
            "valid": ref.valid[0, start:stop].astype(int).tolist(),
            "filtered": ref.filtered[0, start:stop].tolist(),
            "sum": ref.sums[0, bins].tolist(),
            "count": ref.counts[0, bins].tolist(),
        },
    }


# ----------------------------------------------------------- gate level and activity


def gate_level(config: U7Config, stimulus: Stimulus, directory: Path) -> dict[str, Any]:
    """Synthesise to gates, simulate the netlist, compare with the RTL, count toggles.

    The netlist runs on the first ``activity_s`` seconds of the stimulus (or on
    silence), and the RTL model runs on the same codes; their output files must
    be identical.
    """
    spec = config.spec
    gates = tools.synthesise_gates(spec, directory / "gates")
    head = stimulus.head(round(config.activity_s * spec.fs_hz))
    if config.silent:
        head = Stimulus(np.zeros_like(head.codes), head.onsets[:0], head.fs_hz, head.info)
    rtl_program = tools.build_model(spec, directory / "verilator_rtl")
    rtl = tools.run_model(rtl_program, head, spec, directory / "activity_rtl")
    gate_program = tools.build_model(
        spec, directory / "verilator_gates", netlist=Path(gates["netlist"]), activity=True
    )
    saif = directory / "activity.saif"
    gate = tools.run_model(
        gate_program, head, spec, directory / "activity_gates", activity_path=saif
    )
    pairs = (
        (rtl.cancelled_raw, gate.cancelled_raw),
        (rtl.filtered_raw, gate.filtered_raw),
        (rtl.ok, gate.ok),
        (rtl.sums, gate.sums),
        (rtl.counts, gate.counts),
    )
    mismatches = sum(
        int(np.count_nonzero(np.asarray(a) != np.asarray(b))) if a.shape == b.shape else a.size
        for a, b in pairs
    )
    samples = head.n_samples * spec.n_channels
    activity = summarise_activity(tools.read_saif(saif), samples)
    return {
        "gates": {k: v for k, v in gates.items() if k != "netlist"},
        "seconds": head.seconds,
        "silent": config.silent,
        "values_compared": int(sum(a.size for a, _ in pairs)),
        "mismatches_against_rtl": mismatches,
        "window_fraction": 0.0 if config.silent else head.window_fraction(spec.n_taps),
        "simulation_s": float(gate.report["wall_s"]),
        "activity": activity,
    }


# -------------------------------------------------------------------------- synthesis


def _ecp5(spec: ChainSpec, config: U7Config, directory: Path, **options: Any) -> dict[str, Any]:
    """Synthesise one configuration; a failure is recorded, not raised."""
    try:
        row = tools.synthesise_ecp5(
            spec, directory, freq_mhz=config.freq_mhz, seeds=config.pnr_seeds, **options
        )
    except tools.ToolError as error:
        return {"n_channels": spec.n_channels, "error": str(error).splitlines()[-1][:200]}
    row["state_bits"] = spec.n_channels * sum(spec.state_bits.values())
    row["channels_by_timing"] = int(min(row["fmax_mhz"]) * 1e6 / spec.fs_hz)
    row["clock_needed_hz"] = spec.n_channels * spec.fs_hz
    return row


def synthesis(config: U7Config, directory: Path, *, sweep: bool = True) -> dict[str, Any]:
    """Synthesise for the ECP5 across channel counts, and with ``*`` multipliers.

    Returns:
        ``sweep``: one row per channel count. ``variants``: the default channel
        count with the filter's products written as ``*``, mapped to LUTs and
        to the ECP5's hard multipliers.
    """
    spec = config.spec
    counts = config.channel_counts if sweep else (spec.n_channels,)
    if spec.n_channels not in counts:
        counts = tuple(sorted({*counts, spec.n_channels}))
    rows = [_ecp5(replace(spec, n_channels=n), config, directory / f"ecp5_n{n}") for n in counts]
    star = replace(spec, shift_add=False)
    variants = {
        "star_lut": _ecp5(star, config, directory / "ecp5_star_lut"),
        "star_dsp": _ecp5(star, config, directory / "ecp5_star_dsp", dsp=True),
    }
    return {
        "device": tools.DEVICE,
        "package": tools.PACKAGE,
        "speed_grade": tools.SPEED_GRADE,
        "sweep": rows,
        "variants": variants,
    }


# ------------------------------------------------------------------------------ power


def power(config: U7Config, gate: dict[str, Any], stimulus: Stimulus) -> dict[str, Any]:
    """Turn counts and activity into µW per channel and redo the chain's budget.

    The estimate is recomputed with each assumed constant a tenth and ten times
    its value: the constants are orders of magnitude, so the conclusion has to
    hold across that range (U3's rule).
    """
    spec, model = config.spec, config.energy
    window_fraction = stimulus.window_fraction(spec.n_taps)
    measured = {
        "toggles_per_sample": gate["activity"]["toggles_per_sample"],
        "cycles_per_sample": gate["activity"]["cycles_per_sample"],
        "flip_flops": gate["gates"]["flip_flops"],
        "gates": gate["gates"]["gates"],
        "memory_bits_per_sample": spec.memory_bits_per_sample(window_fraction),
        "memory_bits": gate["gates"]["memory_bits"],
        "n_channels": spec.n_channels,
        "fs_hz": spec.fs_hz,
    }
    central = chain_energy(model, **measured)
    scaled = {
        "node capacitance": ("c_node_f",),
        "memory energy per bit": ("memory_j_per_bit",),
        "leakage": ("gate_leakage_w", "bit_leakage_w"),
        "all three": ("c_node_f", "memory_j_per_bit", "gate_leakage_w", "bit_leakage_w"),
    }
    sensitivity = {}
    for label, names in scaled.items():
        low = replace(model, **{n: getattr(model, n) * 0.1 for n in names})
        high = replace(model, **{n: getattr(model, n) * 10.0 for n in names})
        sensitivity[label] = {
            "tenth_uw": chain_energy(low, **measured)["total_uw"],
            "tenfold_uw": chain_energy(high, **measured)["total_uw"],
        }

    chain_config = ChainConfig("sbp", config.sbp_current_a, spec.code_bits, spec.fs_hz, spec.bin_s)
    modelled = chain_power(chain_config, config.power).as_microwatts()
    multiply_j = config.power.energy_per_multiply_j
    canceller_uw = CANCELLER_MODEL_ADDS * multiply_j * spec.fs_hz * 1e6
    placeholder_uw = modelled["dsp_uw"] + canceller_uw
    operations = chain_config.multiplies_per_sample + CANCELLER_MODEL_ADDS
    word_bits = spec.sum_bits + spec.count_bits
    link_uw = config.power.energy_per_bit_j * word_bits / spec.bin_s * 1e6
    before = modelled["total_uw"] + canceller_uw
    analog = modelled["lna_uw"] + modelled["adc_uw"]

    def total(digital_uw: float) -> float:
        return analog + digital_uw + link_uw

    return {
        "assumptions": asdict(model) | {"toggle_fj": model.toggle_j * 1e15},
        "measured": measured | {"window_fraction": window_fraction},
        "energy": central,
        "sensitivity": sensitivity,
        "placeholder": {
            "filter_uw": modelled["dsp_uw"],
            "canceller_uw": canceller_uw,
            "total_uw": placeholder_uw,
            "pj_per_operation": multiply_j * 1e12,
            "operations_per_sample": operations,
        },
        "implied_pj_per_operation": central["dynamic_pj"] / operations,
        "cross_check": {
            "adders_per_sample": spec.adders_per_sample,
            "pj_per_add": HOROWITZ_ADD_J * 1e12,
            "arithmetic_pj": spec.adders_per_sample * HOROWITZ_ADD_J * 1e12,
            "measured_logic_pj": central["logic_pj"],
        },
        "radio": {
            "bits_per_bin_model": chain_config.bit_rate_bps * spec.bin_s,
            "bits_per_bin_rtl": word_bits,
            "link_model_uw": modelled["link_uw"],
            "link_rtl_uw": link_uw,
        },
        "budget": {
            "lna_uw": modelled["lna_uw"],
            "adc_uw": modelled["adc_uw"],
            "before_uw": before,
            "after_uw": total(central["total_uw"]),
            "after_low_uw": total(sensitivity["all three"]["tenth_uw"]),
            "after_high_uw": total(sensitivity["all three"]["tenfold_uw"]),
            "digital_share": central["total_uw"] / total(central["total_uw"]),
            "lna_share": modelled["lna_uw"] / total(central["total_uw"]),
        },
    }


# ---------------------------------------------------------------------------- compute


def _key(summary: dict[str, Any], spec: ChainSpec) -> dict[str, Any]:
    """Collect the numbers the README table and the key-number script quote."""
    real = summary["bit_exact"]
    key: dict[str, Any] = {
        "bit_exact": real["total_mismatches"] == 0 and real["tag_errors"] == 0,
        "mismatches": real["total_mismatches"],
        "values_compared": real["values_compared"],
        "seconds": real["seconds"],
        "n_channels": spec.n_channels,
    }
    rows = [r for r in summary["synthesis"]["sweep"] if r["n_channels"] == spec.n_channels]
    if rows and "error" not in rows[0]:
        row = rows[0]
        key |= {
            "lut4": row["lut4"],
            "logic_cells": row["logic_cells"],
            "flip_flops": row["flip_flops"],
            "block_rams": row["block_rams"],
            "fmax_mhz": row["fmax_median_mhz"],
            "fmax_min_mhz": min(row["fmax_mhz"]),
            "clock_needed_hz": row["clock_needed_hz"],
            "channels_by_timing": row["channels_by_timing"],
        }
    key |= {
        "digital_uw": summary["power"]["energy"]["total_uw"],
        "digital_low_uw": summary["power"]["sensitivity"]["all three"]["tenth_uw"],
        "digital_high_uw": summary["power"]["sensitivity"]["all three"]["tenfold_uw"],
        "placeholder_uw": summary["power"]["placeholder"]["total_uw"],
        "chain_before_uw": summary["power"]["budget"]["before_uw"],
        "chain_after_uw": summary["power"]["budget"]["after_uw"],
    }
    return key


def compute_u7(
    config: U7Config | None = None,
    stimulus: Stimulus | None = None,
    *,
    sweep: bool = True,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Run the whole U7 analysis and return a JSON-ready summary.

    Args:
        config: Analysis parameters.
        stimulus: Codes to run on; defaults to the Neuralink clips
            (:func:`neurochain.rtl.stimulus.neuralink_stimulus`).
        sweep: Synthesise every channel count of the configuration, not only
            the chain's own.
        directory: Build directory; defaults to ``build/u7``. Simulation
            outputs written there are removed at the end.

    Raises:
        ToolError: If Verilator, Yosys or nextpnr is missing or fails.
    """
    config = config or U7Config()
    spec = config.spec
    directory = directory or tools.build_dir()
    for tool in ("verilator", "yosys", "nextpnr-ecp5"):
        if tools.find_tool(tool) is None:
            raise tools.ToolError(f"{tool} is not installed (see docs/U7_design.md, section 2)")
    if stimulus is None:
        stimulus = neuralink_stimulus(spec, config.seconds, config.seed)
    real, out = bit_exact(stimulus, spec, directory, name="real", max_workers=config.max_workers)
    example = example_traces(stimulus, out, spec, config.example_s)
    worst, _ = bit_exact(
        worst_case_stimulus(spec, seed=config.seed), spec, directory, name="worst", max_workers=1
    )
    gate = gate_level(config, stimulus, directory)
    summary: dict[str, Any] = {
        "config": asdict(config),
        "tools": tools.tool_versions(),
        "parameters": spec.verilog_parameters(),
        "state_bits": spec.state_bits | {"total": sum(spec.state_bits.values())},
        "stimulus": stimulus.info,
        "bit_exact": real,
        "worst_case": worst,
        "widths": width_table(spec, real, worst),
        "gate_level": gate,
        "synthesis": synthesis(config, directory, sweep=sweep),
        "power": power(config, gate, stimulus),
        "example": example,
    }
    summary["key"] = _key(summary, spec)
    del out
    for path in directory.glob("*.bin"):
        path.unlink()
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)
    return summary
