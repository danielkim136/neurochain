"""U5 figures: artifact size, recovery, cancellation on a real recording, decoding.

Every number drawn comes from the summary written to ``results/u5_metrics.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from neurochain import viz

CHAIN_COLORS = {"sbp": viz.COLOR_SPIKE, "crossings": viz.COLOR_LFP}
CHAIN_LABELS = {
    "sbp": "0.48 µW chain (SBP, 300 Hz to 1 kHz)",
    "crossings": "conventional chain (crossings, 1 Hz to 7.5 kHz)",
}
CHAIN_SHORT = {"sbp": "0.48 µW chain", "crossings": "conventional chain"}
TREATMENT_COLORS = {
    "none": viz.INK_SECONDARY,
    "blank": viz.COLOR_LFP,
    "template": viz.COLOR_SPIKE,
}
TREATMENT_LABELS = {
    "none": "no treatment",
    "blank": "blanking",
    "template": "template canceller",
}
DECODER_LABELS = {"wiener": "Wiener filter", "gru": "GRU"}


def _limit_lines(ax: Axes, adc_mv: float, amplifier_mv: float, noise_mv: float) -> None:
    for level, text in (
        (amplifier_mv, f"amplifier rails ({amplifier_mv:.0f} mV)"),
        (adc_mv, f"ADC range ({adc_mv:.0f} mV)"),
        (noise_mv, f"in-vivo noise ({noise_mv * 1000:.0f} µV)"),
    ):
        ax.axhline(level, color=viz.INK_SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.annotate(
            text, (0.0, level), xycoords=("axes fraction", "data"), xytext=(4, 3),
            textcoords="offset points", ha="left", fontsize=7.8, color=viz.INK_SECONDARY,
        )  # fmt: skip


def plot_artifact(summary: dict[str, Any], path: Path) -> Path:
    """Draw artifact amplitude against distance, and across the array that stimulates."""
    prop = summary["propagation"]
    noise_mv = summary["config"]["noise_uv"] / 1000.0
    scenarios = prop["scenarios"]
    same = scenarios[0]
    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 5.6))
        grid = fig.add_gridspec(
            1, 2, left=0.075, right=0.945, top=0.82, bottom=0.22, wspace=0.24,
            width_ratios=(1.25, 1.0),
        )  # fmt: skip
        ax = fig.add_subplot(grid[0, 0])
        distance = np.array(prop["distance_mm"])
        ax.loglog(distance, prop["monopolar_mv"], color=viz.SERIES[0], label="monopolar (1/r)")
        ax.loglog(
            distance,
            prop["bipolar_mv"],
            color=viz.SERIES[1],
            label=f"bipolar, return {prop['pair_separation_um']:.0f} µm away (1/r² far off)",
        )
        _limit_lines(ax, prop["adc_range_mv"], prop["amplifier_range_mv"], noise_mv)
        for scenario in scenarios[1:]:
            ax.plot(
                scenario["distance_mm"], scenario["median_mv"], "o", color=viz.SERIES[0],
                markeredgecolor=viz.SURFACE, markersize=7, zorder=4,
            )  # fmt: skip
            ax.annotate(
                scenario["name"], (scenario["distance_mm"], scenario["median_mv"]),
                xytext=(6, 7), textcoords="offset points", fontsize=8,
            )  # fmt: skip
        ax.set_xlabel("Distance from the stimulating electrode (mm)")
        ax.set_ylabel("Artifact at the electrode (mV)")
        ax.set_xlim(distance[0], distance[-1])
        ax.set_ylim(5e-3, 200)
        ax.legend(loc="upper right", fontsize=8.5)
        viz.panel_label(ax, f"A {prop['current_ua']:.0f} µA pulse in uniform tissue")

        ax = fig.add_subplot(grid[0, 1])
        values = np.array(same["amplitudes_mv"], dtype=float)
        side = round(np.sqrt(values.size + 4))
        full = np.full(side * side, np.nan)
        corners = {0, side - 1, side * (side - 1), side * side - 1}
        full[[i for i in range(side * side) if i not in corners]] = values
        image = full.reshape(side, side)
        stim = np.unravel_index(int(np.nanargmax(image)), image.shape)
        shown = image.copy()
        shown[stim] = np.nan
        cmap = LinearSegmentedColormap.from_list("artifact", [viz.SURFACE, viz.COLOR_LFP])
        mesh = ax.imshow(
            shown, cmap=cmap, norm=LogNorm(vmin=np.nanmin(shown) * 0.7, vmax=np.nanmax(shown))
        )
        for (row, col), value in np.ndenumerate(shown):
            if np.isfinite(value):
                ax.text(
                    col, row, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                    color=viz.INK if value < 20 else viz.SURFACE,
                )  # fmt: skip
        ax.plot(stim[1], stim[0], marker="x", color=viz.COLOR_SPIKE, markersize=9, mew=2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label="Artifact (mV)")
        bar.set_ticks([7, 10, 20, 40], labels=["7", "10", "20", "40"])
        bar.ax.minorticks_off()
        viz.panel_label(ax, "Across the stimulating array (× stimulates)")
        viz.add_headline(
            fig,
            f"A {prop['current_ua']:.0f} µA pulse puts {same['min_mv']:.0f} to "
            f"{same['max_mv']:.0f} mV on every other electrode of its own array",
            f"The ADC takes ±{prop['adc_range_mv']:.0f} mV and the amplifier "
            f"±{prop['amplifier_range_mv']:.0f} mV. {same['share_above_amplifier']:.0%} of the "
            f"array is past the amplifier's rails; {scenarios[-1]['distance_mm']:.0f} mm away the "
            f"artifact is still {scenarios[-1]['median_mv']:.1f} mV.",
        )
        viz.add_provenance(
            fig,
            dataset="none (model): point current sources in uniform resistive tissue, "
            f"conductivity {prop['sigma_s_per_m']:g} S/m (assumed), distant return and reference",
            split="not applicable",
            units="peak potential of the first phase at the electrode, mV; 10 by 10 array at "
            "400 µm pitch, corners unwired",
        )
        return viz.save_figure(fig, path)


def plot_recovery(summary: dict[str, Any], path: Path) -> Path:
    """Draw each chain's response to one pulse and its recovery time against amplitude."""
    rec = summary["recovery"]
    threshold = rec["threshold_uv"]
    amplitudes = np.array(rec["amplitudes_mv"])
    key = summary["key"]
    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 5.6))
        grid = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.82, bottom=0.22, wspace=0.22)
        ax = fig.add_subplot(grid[0, 0])
        ax.axvspan(0.0, rec["pulse_ms"], color=viz.NEUTRAL_FILL, alpha=0.5, linewidth=0)
        for name, chain in rec["chains"].items():
            example = chain["example"]
            t = np.array(example["time_ms"])
            ax.semilogy(
                t, np.abs(example["model_uv"]) + 1e-3, color=CHAIN_COLORS[name],
                label=CHAIN_SHORT[name],
            )  # fmt: skip
            if example["spice_uv"] is not None:
                ax.semilogy(
                    t[::6], np.abs(example["spice_uv"])[::6] + 1e-3, "o", markersize=3.2,
                    markerfacecolor=viz.SURFACE, markeredgecolor=CHAIN_COLORS[name],
                    markeredgewidth=0.9,
                )  # fmt: skip
        ax.axhline(threshold, color=viz.INK_SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.axhline(1000.0, color=viz.INK_SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(7.9, threshold * 1.25, f"recovered: {threshold:.0f} µV", ha="right", fontsize=7.8,
                color=viz.INK_SECONDARY)  # fmt: skip
        ax.text(7.9, 1250.0, "ADC range", ha="right", fontsize=7.8, color=viz.INK_SECONDARY)
        ax.text(rec["pulse_ms"] / 2, 0.14, "pulse", ha="center", fontsize=7.8,
                color=viz.INK_SECONDARY)  # fmt: skip
        ax.set_xlim(0, 8)
        ax.set_ylim(0.1, 4e4)
        ax.set_xlabel("Time since pulse onset (ms)")
        ax.set_ylabel("|Amplifier output|, referred to the input (µV)")
        ax.legend(loc="upper right", fontsize=8.5, title="line: model, circles: ngspice",
                  title_fontsize=7.8)  # fmt: skip
        viz.panel_label(ax, f"One pulse, {key['example_mv']:g} mV at the electrode")

        ax = fig.add_subplot(grid[0, 1])
        for name, chain in rec["chains"].items():
            color = CHAIN_COLORS[name]
            ax.semilogx(amplitudes, chain["model_ms"], color=color, label=CHAIN_LABELS[name])
            if rec["spice"]:
                ax.semilogx(
                    amplitudes, chain["spice_ms"], "o", markersize=5.5, markerfacecolor=viz.SURFACE,
                    markeredgecolor=color, markeredgewidth=1.2,
                )  # fmt: skip
                ax.semilogx(amplitudes, chain["spice_electrode_ms"], "x", color=color, markersize=6)
        ax.plot([], [], "o", markerfacecolor=viz.SURFACE, markeredgecolor=viz.INK_SECONDARY,
                label="ngspice, ideal source")  # fmt: skip
        ax.plot([], [], "x", color=viz.INK_SECONDARY, label="ngspice, electrode in series")
        ax.axvline(summary["propagation"]["adc_range_mv"], color=viz.GRID, linewidth=1.2)
        ax.axvline(summary["propagation"]["amplifier_range_mv"], color=viz.GRID, linewidth=1.2)
        ax.set_xlabel("Artifact at the electrode (mV)")
        ax.set_ylabel(f"Time after the pulse to stay within ±{threshold:.0f} µV (ms)")
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=8)
        viz.panel_label(ax, "Recovery time against artifact size")
        gaps = [c["largest_model_spice_gap_us"] for c in rec["chains"].values()]
        agreement = (
            f"The Python model and ngspice agree within {max(gaps):.0f} µs."
            if rec["spice"]
            else "ngspice was not available for this run."
        )
        viz.add_headline(
            fig,
            f"After a {key['example_mv']:g} mV artifact the 0.48 µW chain needs "
            f"{key['recovery_ms']['sbp']:.1f} ms, the conventional chain "
            f"{key['recovery_ms']['crossings']:.1f} ms",
            "Its 300 Hz high-pass corner is what makes it cheap, and what makes it ring. "
            + agreement,
        )
        viz.add_provenance(
            fig,
            dataset="none (circuit model): capacitive-feedback amplifier with a tanh input pair "
            "and rail clamps, in ngspice and in neurochain.frontend.saturation",
            split="not applicable",
            units="output divided by the gain at 1 kHz, µV; recovery measured from the end of "
            f"the second phase; pulse lasts {rec['pulse_ms']:.1f} ms",
        )
        return viz.save_figure(fig, path)


def plot_cancellation(summary: dict[str, Any], path: Path) -> Path:
    """Draw the treatments on a real recording and their effect on SBP by amplitude."""
    real = summary["in_vivo"]
    sig = summary["signal"]
    amplitudes = np.array(sig["amplitudes_mv"])
    key = summary["key"]
    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 9.0))
        grid = fig.add_gridspec(
            3, 2, left=0.085, right=0.985, top=0.885, bottom=0.135, wspace=0.22, hspace=0.55,
            height_ratios=(0.8, 1.0, 1.25),
        )  # fmt: skip
        if real is not None:
            trace = real["trace"]
            t = np.array(trace["time_ms"])
            ax = fig.add_subplot(grid[0, :])
            ax.step(t, trace["raw_uv"], where="mid", color=viz.INK_SECONDARY, linewidth=1.3)
            for onset in trace["onsets_ms"]:
                ax.axvline(onset, color=viz.GRID, linewidth=1.2)
            ax.set_ylabel("ADC output (µV)")
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelbottom=False)
            viz.panel_label(
                ax,
                f"Real recording with a {real['amplitude_mv']:g} mV artifact injected: what the "
                "ADC delivers",
            )
            ax = fig.add_subplot(grid[1, :])
            ax.plot(t, trace["clean_uv"], color=viz.INK, linewidth=1.1, label="without stimulation")
            for treatment in ("blank", "template"):
                values = np.array(trace[f"{treatment}_uv"], dtype=float)
                values[~np.array(trace[f"{treatment}_valid"])] = np.nan
                ax.plot(
                    t, values, "o-", color=TREATMENT_COLORS[treatment], markersize=3.2,
                    linewidth=1.1, label=TREATMENT_LABELS[treatment] + " (gaps: samples discarded)",
                )  # fmt: skip
            for onset in trace["onsets_ms"]:
                ax.axvline(onset, color=viz.GRID, linewidth=1.2)
            ax.set_xlim(t[0], t[-1])
            ax.set_xlabel("Time (ms); vertical lines are pulse onsets")
            ax.set_ylabel("After treatment (µV)")
            low = min(np.min(trace["clean_uv"]), np.nanmin(values))
            ax.set_ylim(low - 38.0, None)
            ax.legend(loc="lower left", ncols=3, fontsize=7.8)
            viz.panel_label(ax, "The same stretch after each treatment")

        ax = fig.add_subplot(grid[2, 0])
        for treatment in TREATMENT_COLORS:
            ax.plot(
                amplitudes, sig["sbp"][treatment]["bias"], "o-", color=TREATMENT_COLORS[treatment],
                markersize=4.5, label=TREATMENT_LABELS[treatment],
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1, linscale=1.0)
        ax.axhline(0.0, color=viz.INK_SECONDARY, linewidth=0.8)
        ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 1, 10])
        ax.set_yticklabels(["-0.1", "-0.05", "0", "0.05", "0.1", "1", "10"])
        ax.set_xlabel("Artifact at the electrode (mV)")
        ax.set_ylabel("Shift of SBP in stimulated bins (noise floors)")
        ax.legend(loc="upper left", fontsize=8.5)
        viz.panel_label(ax, "What is left in the feature (0.48 µW chain)")

        ax = fig.add_subplot(grid[2, 1])
        for name, style in (("sbp", "-"), ("crossings", (0, (4, 2)))):
            for treatment in ("blank", "template"):
                ax.plot(
                    amplitudes, np.array(sig[name][treatment]["valid"]) * 100, "o",
                    linestyle=style, color=TREATMENT_COLORS[treatment], markersize=4.5,
                    label=f"{TREATMENT_LABELS[treatment]}, {CHAIN_SHORT[name]}",
                )  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(50, 102)
        ax.set_xlabel("Artifact at the electrode (mV)")
        ax.set_ylabel("Samples kept while stimulating (%)")
        ax.legend(loc="lower left", fontsize=7.8)
        viz.panel_label(ax, f"What survives at {sig['realised_rate_hz']:.0f} pulses per second")
        viz.add_headline(
            fig,
            f"Subtracting a learned template keeps {key['valid']['template']:.0%} of the samples "
            f"at {key['example_mv']:g} mV; blanking keeps {key['valid']['blank']:.0%}",
            "Untreated, the artifact raises SBP by "
            f"{key['bias']['none']:.0f} noise floors. Both treatments bring that within "
            f"{max(abs(key['bias']['blank']), abs(key['bias']['template'])):.2f}.",
        )
        viz.add_provenance(
            fig,
            dataset="top: Nason et al. in-vivo broadband (one channel, 10 s); bottom: MEArec "
            f"simulation, {sig['n_channels']} channels, {summary['config']['noise_uv']:g} µV "
            "background, artifact injected by the amplifier model",
            split=f"stimulated bins only ({sig['stimulated_s']:.1f} s in {sig['n_pulses']} pulses "
            "on MEArec); no fitting",
            units="µV at the electrode; SBP shift in multiples of each channel's noise-only SBP; "
            "50 ms bins",
        )
        return viz.save_figure(fig, path)


def _rows(summary: dict[str, Any], **match: str) -> list[dict[str, Any]]:
    return [
        r
        for r in summary["decoding"]["rows"]
        if all(r[key] == value for key, value in match.items())
    ]


def plot_decoding(summary: dict[str, Any], path: Path) -> Path:
    """Draw R² on stimulated bins for each geometry, chain, treatment, decoder and fit."""
    decode = summary["decoding"]
    key = summary["key"]
    decoders = list(dict.fromkeys(r["decoder"] for r in decode["rows"]))[::-1]
    scenarios = list(dict.fromkeys(r["scenario"] for r in decode["rows"]))
    series = [(c, t) for c in CHAIN_COLORS for t in ("blank", "template")]
    offsets = np.linspace(-0.27, 0.27, len(series))
    fits = ["without stimulation", "with stimulation"]

    def draw(ax: Axes, rows: list[dict[str, Any]], labels: list[str], field: str) -> None:
        decoder = rows[0]["decoder"]
        for (chain, treatment), offset in zip(series, offsets, strict=True):
            chosen = {
                r[field]: r for r in rows if r["chain"] == chain and r["treatment"] == treatment
            }
            x = np.array([labels.index(label) + offset for label in chosen])
            y = np.array([r["r2"] for r in chosen.values()])
            low = np.array([r["r2_low"] for r in chosen.values()])
            high = np.array([r["r2_high"] for r in chosen.values()])
            filled = treatment == "template"
            ax.errorbar(
                x, y, yerr=[y - low, high - y], fmt="o", color=CHAIN_COLORS[chain], markersize=6,
                markerfacecolor=CHAIN_COLORS[chain] if filled else viz.SURFACE,
                markeredgecolor=CHAIN_COLORS[chain], markeredgewidth=1.4, elinewidth=1.5,
                label=f"{CHAIN_SHORT[chain]}, {TREATMENT_LABELS[treatment]}", zorder=3,
            )  # fmt: skip
        reference = decode["no_stimulation_r2"]
        for chain, color in CHAIN_COLORS.items():
            ax.axhline(reference[chain][decoder], color=color, linewidth=1.0, linestyle=(0, (4, 3)))
        top = max(reference[chain][decoder] for chain in CHAIN_COLORS)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(top - 0.45, top + 0.07)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"R², stimulated bins ({DECODER_LABELS.get(decoder, decoder)})")

    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 9.4))
        grid = fig.add_gridspec(
            len(decoders), 2, left=0.08, right=0.985, top=0.855, bottom=0.14, wspace=0.2,
            hspace=0.42, width_ratios=(1.4, 1.0),
        )  # fmt: skip
        for slot, decoder in enumerate(decoders):
            ax = fig.add_subplot(grid[slot, 0])
            draw(ax, _rows(summary, decoder=decoder, fitted=fits[0]), scenarios, "scenario")
            viz.panel_label(
                ax, f"{DECODER_LABELS.get(decoder, decoder)}, fitted without stimulation"
            )
            if slot == len(decoders) - 1:
                ax.legend(loc="lower left", ncols=2, fontsize=7.8)
                ax.set_xlabel("Where the stimulating electrode is")
            ax = fig.add_subplot(grid[slot, 1])
            draw(ax, _rows(summary, decoder=decoder, scenario=scenarios[0]), fits, "fitted")
            ax.set_ylabel("")
            ax.set_xticklabels(["fitted without\nstimulation", "refitted with\nstimulation on"])
            viz.panel_label(ax, f"{scenarios[0].capitalize()}: what refitting recovers")

        def row(decoder: str, treatment: str, fitted: str, chain: str = "sbp") -> dict[str, Any]:
            return _rows(
                summary, decoder=decoder, scenario=scenarios[0], fitted=fitted, chain=chain,
                treatment=treatment,
            )[0]  # fmt: skip

        refit = row(key["decoder"], "template", fits[1])
        linear = decoders[-1]
        worst = [key[f"{c}_none"]["r2"] for c in CHAIN_COLORS]
        viz.add_headline(
            fig,
            "While its own array stimulates, the 0.48 µW chain with a canceller decodes at "
            f"R² {refit['r2']:.2f}",
            f"GRU refitted with stimulation on: {refit['loss']:+.3f} [{refit['loss_low']:+.3f}, "
            f"{refit['loss_high']:+.3f}] against no stimulation; not refitted "
            f"{key['sbp_template']['loss']:+.3f}; blanking instead {key['sbp_blank']['loss']:+.3f}.",
        )
        fig.text(
            0.012, 0.918,
            f"With the {DECODER_LABELS.get(linear, linear)} the same chain loses "
            f"{-row(linear, 'template', fits[1])['loss']:.3f} even refitted; the conventional "
            f"chain with blanking loses {-row(linear, 'blank', fits[1], 'crossings')['loss']:.3f}. "
            f"Untreated: R² {min(worst):.2f} (GRU).",
            ha="left", va="top", fontsize=9.5, color=viz.INK_SECONDARY,
        )  # fmt: skip
        viz.add_provenance(
            fig,
            dataset="Nason et al. online data (96 channels, finger velocity, 50 ms bins); "
            "stimulation effect measured on MEArec and applied to the recorded features",
            split=f"5 contiguous blocks of trials held out in turn; scored on the "
            f"{decode['stimulated_bins']} stimulated bins of {decode['n_bins']}; 95 % intervals "
            "from resampling trials",
            units="R² of decoded finger velocity; artifact amplitude per channel from the "
            "point-source model; dashed lines: no stimulation",
        )
        return viz.save_figure(fig, path)
