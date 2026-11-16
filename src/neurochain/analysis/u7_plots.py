"""U7 figures: RTL against Python, ECP5 resources against channels, the power estimate.

Every number drawn comes from the summary written to ``results/u7_metrics.json``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neurochain import viz

WEAK = "#a3a29c"


def _headline(fig: Figure, headline: str, subtitle: str) -> None:
    """Add the headline and a subtitle wrapped to the figure's width."""
    width = round(fig.get_figwidth() * 12.6)
    viz.add_headline(fig, headline, "\n".join(textwrap.wrap(subtitle, width)))


def _provenance(fig: Figure, *, dataset: str, split: str, units: str) -> None:
    """Add the provenance block with each entry wrapped to the figure's width."""
    width = round(fig.get_figwidth() * 15.0)

    def wrap(text: str) -> str:
        return "\n".join(textwrap.wrap(text, width, subsequent_indent="    "))

    viz.add_provenance(fig, dataset=wrap(dataset), split=wrap(split), units=wrap(units))


def _dataset(summary: dict[str, Any]) -> str:
    """Describe the codes the comparison ran on."""
    stim = summary["stimulus"]
    if stim.get("source") != "neuralink":
        return f"{stim.get('source', 'unknown')} codes, {stim['n_channels']} channels"
    low, high = stim["band_hz"]
    amplitudes = stim["artifact_amplitudes_mv"]
    return (
        f"Neuralink compression-challenge clips ({stim['n_clips']} clips, "
        f"{stim['unique_seconds'] / 60:.0f} min, 10-bit, 19.5 kS/s, no voltage scale): causal "
        f"{low:g} Hz to {high / 1000:g} kHz band-pass, sampled at 2 kS/s, divided by 4 to 8-bit "
        f"codes; each of {stim['n_channels']} channels a seeded shuffle of the clips (seed "
        f"{stim['seed']}); U5's stimulation artifacts added before rounding, "
        f"{amplitudes[0]:g} to {amplitudes[-1]:g} mV at {stim['pulse_rate_hz']:.0f} pulses per "
        f"second, a new size per train and channel"
    )


# ---------------------------------------------------------------------- bit-exactness


def _overlay(ax: Axes, t: np.ndarray, python: np.ndarray, rtl: np.ndarray) -> None:
    ax.plot(t, python, color=viz.COLOR_LFP, linewidth=2.6, alpha=0.45, label="Python reference")
    ax.plot(
        t, rtl, linestyle="none", marker="o", markersize=1.9, color=viz.COLOR_SPIKE,
        label="RTL (Verilator)",
    )  # fmt: skip


def plot_bit_exact(summary: dict[str, Any], path: Path) -> Path:
    """Draw one stretch of real data through both chains, and the word widths."""
    real, example, widths = summary["bit_exact"], summary["example"], summary["widths"]
    scale = example["scale"]
    t = np.arange(len(example["codes"])) / example["fs_hz"] * 1000.0
    rtl, python = example["rtl"], example["python"]
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 7.8))
        left, width = 0.065, 0.52
        ax_code = fig.add_axes((left, 0.685, width, 0.145))
        ax_cancel = fig.add_axes((left, 0.48, width, 0.145), sharex=ax_code)
        ax_filter = fig.add_axes((left, 0.275, width, 0.145), sharex=ax_code)
        ax_bin = fig.add_axes((0.665, 0.645, 0.32, 0.185))
        ax_width = fig.add_axes((0.735, 0.275, 0.25, 0.25))

        ax_code.step(t, example["codes"], where="mid", color=viz.INK, linewidth=0.9)
        for onset in example["onsets"]:
            ax_code.axvline(t[onset], color=viz.GRID, linewidth=0.6, zorder=0)
        viz.panel_label(ax_code, f"ADC codes, channel {example['channel']} (grey: pulse onsets)")
        ax_code.set_ylabel("code (LSB)")

        _overlay(
            ax_cancel, t, np.array(python["cancelled"]) / scale, np.array(rtl["cancelled"]) / scale
        )
        blanked = np.flatnonzero(np.array(rtl["valid"]) == 0)
        ax_cancel.plot(
            t[blanked], np.zeros(blanked.size), linestyle="none", marker="x", markersize=3.5,
            color=viz.INK_SECONDARY, label="blanked",
        )  # fmt: skip
        viz.panel_label(ax_cancel, "After the template canceller")
        ax_cancel.set_ylabel("LSB")
        ax_cancel.legend(loc="upper right", fontsize=7.6, ncols=3)

        _overlay(
            ax_filter, t, np.array(python["filtered"]) / scale, np.array(rtl["filtered"]) / scale
        )
        viz.panel_label(ax_filter, "After the 300 Hz high-pass")
        ax_filter.set_ylabel("LSB")
        ax_filter.set_xlabel(f"time from {example['start_s']:.2f} s (ms)")
        for ax in (ax_code, ax_cancel):
            plt.setp(ax.get_xticklabels(), visible=False)

        bins = np.arange(len(python["sum"]))
        ax_bin.bar(bins, python["sum"], width=0.62, color=viz.COLOR_LFP, alpha=0.45, label="Python")
        ax_bin.plot(
            bins, rtl["sum"], linestyle="none", marker="o", markersize=4.0, color=viz.COLOR_SPIKE,
            label="RTL",
        )  # fmt: skip
        for i, count in enumerate(rtl["count"]):
            ax_bin.annotate(
                str(count), (i, rtl["sum"][i]), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=7.2, color=viz.INK_SECONDARY,
            )  # fmt: skip
        ax_bin.set_ylim(0, max(python["sum"]) * 1.22)
        ax_bin.set_xticks(bins)
        ax_bin.set_xlabel("50 ms bin of the stretch")
        ax_bin.set_ylabel("sum of |y| (quarter LSB)")
        viz.panel_label(ax_bin, "The feature (number: samples kept)")
        ax_bin.legend(loc="upper right", fontsize=7.6)

        rows = np.arange(len(widths))[::-1]
        ax_width.barh(
            rows, [w["bits"] for w in widths], height=0.62, color=viz.NEUTRAL_FILL, label="register"
        )
        ax_width.barh(
            rows, [w["bits_used_real"] for w in widths], height=0.3, color=viz.COLOR_LFP,
            label="used by this data",
        )  # fmt: skip
        for row, w in zip(rows, widths, strict=True):
            ax_width.text(w["bits"] + 0.4, row, f"{w['bits']}", va="center", fontsize=8.0)
        ax_width.set_yticks(rows, [w["node"] for w in widths])
        ax_width.set_xlim(0, max(w["bits"] for w in widths) + 9.0)
        ax_width.set_xlabel("bits")
        ax_width.grid(axis="y", visible=False)
        viz.panel_label(ax_width, "Word widths: worst case")
        ax_width.legend(loc="upper right", fontsize=7.4)

        minutes = real["seconds"] / 60.0
        if real["total_mismatches"] == 0:
            headline = (
                f"RTL equals the Python reference on all {real['values_compared']:,} values: "
                "zero mismatches"
            )
        else:
            headline = (
                f"RTL differs from Python on {real['total_mismatches']:,} of "
                f"{real['values_compared']:,} values"
            )
        worst, gate = summary["worst_case"], summary["gate_level"]
        _headline(
            fig,
            headline,
            f"{real['n_channels']} channels for {minutes:.0f} minutes: {real['samples_compared']:,} "
            f"samples (canceller output, blanking flag, filter output) and "
            f"{real['bins_compared']:,} bins (sum, count). A stimulus built to reach every "
            f"register's bound: {worst['total_mismatches']} mismatches. The gate-level netlist "
            f"against the RTL on {gate['seconds']:.0f} s: {gate['mismatches_against_rtl']}.",
        )
        _provenance(
            fig,
            dataset=_dataset(summary),
            split=f"no fitting, so no split: all {minutes:.0f} minutes of every channel are "
            f"compared; drawn: {len(t) / example['fs_hz'] * 1000:.0f} ms of channel "
            f"{example['channel']} at the start of a pulse train whose artifact differs from the "
            "one the template last learnt",
            units="ADC codes in LSB of the 8-bit converter (no voltage scale exists for this "
            "data); canceller and filter outputs are held in quarter LSB and drawn in LSB; bin "
            "sums in quarter LSB; widths in bits",
        )
        return viz.save_figure(fig, path)


# -------------------------------------------------------------------------- synthesis


def plot_synthesis(summary: dict[str, Any], path: Path) -> Path:
    """Draw LUTs, block RAM and clock rate against the number of channels."""
    synth = summary["synthesis"]
    rows = [r for r in synth["sweep"] if "error" not in r]
    failed = [r["n_channels"] for r in synth["sweep"] if "error" in r]
    n = np.array([r["n_channels"] for r in rows], dtype=float)
    key = summary["key"]
    single = next((r for r in rows if r["n_channels"] == 1), None)
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 6.0))
        ax_lut = fig.add_axes((0.065, 0.27, 0.26, 0.5))
        ax_ram = fig.add_axes((0.395, 0.27, 0.26, 0.5))
        ax_clk = fig.add_axes((0.725, 0.27, 0.26, 0.5))

        ax_lut.plot(
            n,
            [r["lut4"] for r in rows],
            marker="o",
            markersize=4,
            color=viz.SERIES[0],
            label="one shared datapath",
        )
        ax_lut.plot(
            n,
            [r["flip_flops"] for r in rows],
            marker="o",
            markersize=3,
            color=viz.SERIES[0],
            linewidth=1.0,
            alpha=0.5,
            label="its flip-flops",
        )
        if single is not None:
            ax_lut.plot(
                n,
                n * single["lut4"],
                color=viz.SERIES[1],
                linestyle=(0, (4, 3)),
                label="a datapath per channel",
            )
        limit = rows[0]["logic_cells_available"]
        ax_lut.axhline(limit, color=viz.INK, linewidth=0.9, linestyle=(0, (1, 2)))
        ax_lut.text(
            n[-1],
            limit * 0.5,
            f"{synth['device']}: {limit:,} LUT4",
            fontsize=7.8,
            color=viz.INK_SECONDARY,
            ha="right",
        )
        ax_lut.set_xscale("log")
        ax_lut.set_yscale("log")
        ax_lut.set_ylim(60, limit * 30)
        ax_lut.set_ylabel("cells")
        ax_lut.legend(loc="upper left", fontsize=7.6)
        viz.panel_label(ax_lut, "Logic")

        ax_ram.plot(
            n, [r["block_rams"] for r in rows], marker="o", markersize=4, color=viz.SERIES[0]
        )
        ram_limit = rows[0]["block_rams_available"]
        ax_ram.axhline(ram_limit, color=viz.INK, linewidth=0.9, linestyle=(0, (1, 2)))
        ax_ram.text(
            n[0],
            ram_limit + 1.5,
            f"{ram_limit} block RAMs of 18 kbit",
            fontsize=7.8,
            color=viz.INK_SECONDARY,
        )
        for r in rows:
            if r["block_rams"]:
                ax_ram.annotate(
                    f"{r['state_bits'] / 1000:.0f} kbit", (r["n_channels"], r["block_rams"]),
                    textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=7.4,
                    color=viz.INK_SECONDARY,
                )  # fmt: skip
        ax_ram.set_xscale("log")
        ax_ram.set_ylim(0, ram_limit * 1.18)
        ax_ram.set_ylabel("block RAMs used")
        viz.panel_label(ax_ram, "State (label: bits stored)")

        median = np.array([r["fmax_median_mhz"] for r in rows])
        low = np.array([min(r["fmax_mhz"]) for r in rows])
        high = np.array([max(r["fmax_mhz"]) for r in rows])
        ax_clk.errorbar(
            n, median, yerr=(median - low, high - median), marker="o", markersize=4,
            color=viz.SERIES[0], capsize=2.5, label="fastest clock the design allows",
        )  # fmt: skip
        ax_clk.plot(
            n,
            [r["clock_needed_hz"] / 1e6 for r in rows],
            color=viz.SERIES[1],
            marker="o",
            markersize=3,
            label="clock the channels need",
        )
        ax_clk.set_xscale("log")
        ax_clk.set_yscale("log")
        ax_clk.set_ylim(1e-3, 3e3)
        ax_clk.set_ylabel("MHz")
        ax_clk.legend(loc="upper left", fontsize=7.6)
        viz.panel_label(ax_clk, "Clock")
        for ax in (ax_lut, ax_ram, ax_clk):
            ax.set_xlabel("channels sharing the datapath")

        star = synth["variants"]
        path_info = next(r for r in rows if r["n_channels"] == key["n_channels"])["critical_path"]
        note = f" Not built: {', '.join(map(str, failed))} channels." if failed else ""
        _headline(
            fig,
            f"{key['n_channels']} channels fit in {key['lut4']} LUTs and {key['block_rams']} block "
            f"RAMs, with {key['fmax_min_mhz'] * 1e6 / key['clock_needed_hz']:.0f} times the "
            "clock rate needed",
            f"{key['flip_flops']} flip-flops, fmax {key['fmax_mhz']:.1f} MHz (slowest of "
            f"{len(rows[0]['seeds'])} placements {key['fmax_min_mhz']:.1f}) against "
            f"{key['clock_needed_hz'] / 1000:.0f} kHz needed. The slowest path leaves the "
            f"{'filter-state RAM' if path_info.get('from') == 'u_flt' else path_info.get('from')}, "
            f"whose output alone takes {path_info.get('clk_to_q_ns', float('nan')):.1f} of "
            f"{path_info.get('delay_ns', float('nan')):.1f} ns. Writing the filter's products as "
            f'"*": {star["star_lut"]["lut4"]} LUTs, or {star["star_dsp"]["lut4"]} plus '
            f"{star['star_dsp']['multipliers']} hard multipliers.{note}",
        )
        _provenance(
            fig,
            dataset=f"rtl/nc_sbp_chain.sv synthesised by Yosys {summary['tools'].get('yosys', '?')} "
            f"(synth_ecp5 -nodsp) and placed and routed by nextpnr-ecp5 for a Lattice "
            f"{synth['device']}, {synth['package']}, speed grade {synth['speed_grade']}; no "
            "hardware in the loop",
            split=f"one build per channel count; clock: median and range of placement seeds "
            f"{rows[0]['seeds']}; dashed orange: channels times the one-channel build's LUTs",
            units="LUT4 and flip-flop counts from Yosys; block RAMs are 18 kbit DP16KD; clock in "
            "MHz from nextpnr's timing analysis of the routed design",
        )
        return viz.save_figure(fig, path)


# ------------------------------------------------------------------------------ power


def plot_power(summary: dict[str, Any], path: Path) -> Path:
    """Draw the energy per sample, its sensitivity, and the chain's budget."""
    power = summary["power"]
    energy, placeholder, check = power["energy"], power["placeholder"], power["cross_check"]
    budget, radio, key = power["budget"], power["radio"], summary["key"]
    assumed = power["assumptions"]
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 6.4))
        ax_energy = fig.add_axes((0.2, 0.61, 0.775, 0.17))
        ax_range = fig.add_axes((0.2, 0.295, 0.3, 0.2))
        ax_budget = fig.add_axes((0.675, 0.295, 0.3, 0.2))

        parts = [
            ("logic", energy["logic_pj"], viz.SERIES_STACK[0]),
            ("clock", energy["clock_pj"], viz.SERIES_STACK[2]),
            ("state memory", energy["memory_pj"], viz.SERIES_STACK[1]),
        ]
        start = 0.0
        for label, value, color in parts:
            ax_energy.barh(2, value, left=start, height=0.6, color=color, label=label)
            ax_energy.text(
                start + value / 2,
                2,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white",
            )
            start += value
        ax_energy.text(start + 0.06, 2, f"{start:.2f} pJ", va="center", fontsize=8.4)
        total_placeholder = placeholder["operations_per_sample"] * placeholder["pj_per_operation"]
        ax_energy.barh(1, total_placeholder, height=0.6, color=viz.NEUTRAL_FILL)
        ax_energy.text(
            total_placeholder + 0.06, 1, f"{total_placeholder:.2f} pJ", va="center", fontsize=8.4
        )
        ax_energy.barh(0, check["arithmetic_pj"], height=0.6, color=viz.NEUTRAL_FILL)
        ax_energy.text(
            check["arithmetic_pj"] + 0.06,
            0,
            f"{check['arithmetic_pj']:.2f} pJ (compare logic + clock)",
            va="center",
            fontsize=8.4,
        )
        ax_energy.set_yticks(
            [2, 1, 0],
            [
                "synthesised, from activity",
                f"placeholder: {placeholder['operations_per_sample']} operations x "
                f"{placeholder['pj_per_operation']:g} pJ",
                f"{check['adders_per_sample']} additions x {check['pj_per_add']:g} pJ",
            ],
        )
        ax_energy.set_xlim(0, max(start, total_placeholder) * 1.3)
        ax_energy.set_xlabel("energy per sample of one channel (pJ)")
        ax_energy.grid(axis="y", visible=False)
        ax_energy.legend(loc="lower right", fontsize=7.8, ncols=3)
        viz.panel_label(ax_energy, "Dynamic energy per sample")

        labels = list(power["sensitivity"])
        rows = np.arange(len(labels))[::-1]
        for row, label in zip(rows, labels, strict=True):
            span = power["sensitivity"][label]
            ax_range.plot(
                [span["tenth_uw"], span["tenfold_uw"]],
                [row, row],
                color=viz.SERIES[0],
                linewidth=4.0,
                solid_capstyle="butt",
            )
        ax_range.axvline(energy["total_uw"], color=viz.INK, linewidth=1.0)
        ax_range.axvline(
            placeholder["total_uw"], color=viz.SERIES[1], linewidth=1.0, linestyle=(0, (4, 3))
        )
        ax_range.set_xscale("log")
        ax_range.set_yticks(rows, labels)
        ax_range.set_ylim(-0.6, len(labels) - 0.4)
        ax_range.set_xlabel("digital power (µW per channel)")
        ax_range.grid(axis="y", visible=False)
        viz.panel_label(ax_range, "Each assumption ÷10 to x10")

        stages = [
            ("amplifier", "lna_uw", viz.SERIES_STACK[0]),
            ("ADC", "adc_uw", viz.SERIES_STACK[2]),
            ("digital", "digital", viz.SERIES_STACK[1]),
            ("radio", "radio", viz.SERIES_STACK[3]),
        ]
        bars = {
            "U3 + U5 model": {
                "digital": placeholder["total_uw"],
                "radio": radio["link_model_uw"],
                "total": budget["before_uw"],
            },
            "with U7's estimate": {
                "digital": energy["total_uw"],
                "radio": radio["link_rtl_uw"],
                "total": budget["after_uw"],
            },
        }
        for row, own in zip((1, 0), bars.values(), strict=True):
            start = 0.0
            for name, field, color in stages:
                value = budget[field] if field in budget else own[field]
                ax_budget.barh(
                    row,
                    value,
                    left=start,
                    height=0.55,
                    color=color,
                    label=name if row == 1 else None,
                )
                start += value
            ax_budget.text(start + 0.008, row, f"{own['total']:.3f}", va="center", fontsize=8.4)
        ax_budget.set_yticks([1, 0], list(bars))
        ax_budget.set_xlim(0, budget["after_uw"] * 1.22)
        ax_budget.set_ylim(-0.6, 2.25)
        ax_budget.set_xlabel("chain power (µW per channel)")
        ax_budget.grid(axis="y", visible=False)
        ax_budget.legend(
            loc="upper left", fontsize=7.4, ncols=4, handlelength=1.0, columnspacing=0.9
        )
        viz.panel_label(ax_budget, "The chain's budget")

        _headline(
            fig,
            f"Digital logic: {key['digital_uw']:.3f} µW per channel; the chain moves from "
            f"{budget['before_uw']:.3f} to {budget['after_uw']:.3f} µW",
            f"The placeholder charged {placeholder['total_uw']:.3f} µW. State memory, not "
            f"arithmetic, is {energy['memory_pj'] / energy['dynamic_pj']:.0%} of the dynamic "
            f"energy; leakage adds {energy['leakage_uw']:.4f} µW. With every assumed constant "
            f"between a tenth and ten times its value the chain stays between "
            f"{budget['after_low_uw']:.3f} and {budget['after_high_uw']:.3f} µW, and the "
            f"amplifier is {budget['lna_share']:.0%} of it. An ASIC estimate: an FPGA's own "
            "power is not the implant's.",
        )
        gates = power["measured"]
        _provenance(
            fig,
            dataset=f"activity: Yosys NAND/NOR netlist ({gates['gates']} gates, "
            f"{gates['flip_flops']} flip-flops) simulated by Verilator on the first "
            f"{summary['gate_level']['seconds']:.0f} s of the bit-exactness data (Neuralink clips, "
            f"{gates['n_channels']} channels, stimulation on); "
            f"{gates['toggles_per_sample']:.0f} node toggles and "
            f"{gates['memory_bits_per_sample']:.0f} memory bits per sample",
            split="assumptions, none measured: "
            f"{assumed['c_node_f'] * 1e15:g} fF per node at {assumed['v_supply_v']:g} V "
            f"({assumed['toggle_fj']:.2f} fJ per toggle), "
            f"{assumed['memory_j_per_bit'] * 1e15:g} fJ per memory bit read or written, "
            f"{assumed['gate_leakage_w'] * 1e12:g} pW leakage per gate and per bit (65 nm "
            "low-power process); amplifier, ADC and radio from U3's power model",
            units="pJ per sample of one channel; µW per channel at 2 kS/s; black line: this "
            "estimate; dashed orange: the 1 pJ-per-operation placeholder of U3 and U5; radio word "
            f"{radio['bits_per_bin_model']:.0f} bits per bin in the model, "
            f"{radio['bits_per_bin_rtl']} as built",
        )
        return viz.save_figure(fig, path)
