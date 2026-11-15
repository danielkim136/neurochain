"""U6 figures: the aging model, R² against weeks, the real session pair, robustness.

Every number drawn comes from the summary written to ``results/u6_metrics.json``.
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

DECODER_LABELS = {
    "ridge": "U3's ridge readout",
    "wiener": "Wiener filter",
    "kalman": "Kalman filter",
    "gru": "GRU",
}
CONDITION_TITLES = {
    "nason_sbp": "0.48 µW chain (SBP), Nason",
    "nason_crossings": "19.6 µW chain (crossings), Nason",
    "nomad_session2_crossings": "19.6 µW chain, NoMAD session 2",
}
DATASET_NAMES = {"nason": "Nason et al. online data", "nomad_session2": "NoMAD session 2"}
MODE_COLORS = {
    "fixed": viz.INK_SECONDARY,
    "zscore": viz.SERIES[0],
    "refit": viz.SERIES[2],
    "aligned": viz.SERIES[1],
}
MODE_LABELS = {
    "fixed": "decoder left alone",
    "mean": "running mean",
    "zscore": "running z-score",
    "refit": "refitted with labels",
    "batch": "whole-session z-score",
    "latent": "10 latents, not aligned",
    "aligned": "10 latents, aligned",
    "within": "refitted with labels",
    "gated zscore": "z-score gated by the stimulator",
    "no stimulation": "no stimulation",
}
WEAK = "#a3a29c"


def _rows(rows: list[dict[str, Any]], **match: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def _one(rows: list[dict[str, Any]], **match: Any) -> dict[str, Any]:
    found = _rows(rows, **match)
    if len(found) != 1:
        raise KeyError(f"{len(found)} rows match {match}")
    return found[0]


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


def _title(condition: str) -> str:
    return CONDITION_TITLES.get(condition, condition.replace("_", " "))


def _best_decoder(summary: dict[str, Any], condition: str) -> str:
    """Return the decoder with the highest week-0 score on a condition."""
    curves: dict[str, dict[str, float]] = summary["key"]["curves"][condition]
    return max(curves, key=lambda name: curves[name]["week0"])


# ----------------------------------------------------------------------------- the model


def plot_model(summary: dict[str, Any], path: Path) -> Path:
    """Draw the aging model and what it does to threshold crossings on MEArec."""
    model, response = summary["model"], summary["response"]
    weeks = np.asarray(model["fine_weeks"])
    key_week = summary["key"]["week"] if "key" in summary else summary["config"]["key_week"]
    first, last = model["states"][0], model["states"][-1]
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 5.6))
        grid = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=0.78, bottom=0.29, wspace=0.3)

        ax = fig.add_subplot(grid[0, 0])
        spectrum = model["spectrum"]
        freqs = np.asarray(spectrum["freqs_hz"])
        ax.loglog(freqs, np.asarray(spectrum["fresh_ohm"]) / 1e3, color=WEAK, label="week 0")
        ax.loglog(
            freqs,
            np.asarray(spectrum["settled_ohm"]) / 1e3,
            color=viz.SERIES[1],
            label=f"settled (week {key_week:g})",
        )
        ax.axvline(1000.0, color=viz.GRID, linewidth=1.0, zorder=0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Electrode impedance |Z| (kΩ)")
        ax.legend(loc="upper right", fontsize=8.2)
        viz.panel_label(
            ax, f"Sheath: {first['z_1khz_kohm']:.0f} to {last['z_1khz_kohm']:.0f} kΩ at 1 kHz"
        )

        ax = fig.add_subplot(grid[0, 1])
        amplitude = model["amplitude"]
        ax.fill_between(
            weeks, amplitude["p10"], amplitude["p90"], color=viz.SERIES[0], alpha=0.18, linewidth=0
        )
        ax.plot(weeks, amplitude["p50"], color=viz.SERIES[0], label="median channel")
        ax.plot(weeks, amplitude["mean"], color=viz.SERIES[0], linestyle=(0, (4, 3)), label="mean")
        ax.axvline(key_week, color=viz.GRID, linewidth=1.0, zorder=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Simulated weeks after implantation")
        ax.set_ylabel("Spike amplitude relative to week 0")
        ax.legend(loc="lower left", fontsize=8.2)
        viz.panel_label(ax, "Spikes shrink, each channel at its rate")

        ax = fig.add_subplot(grid[0, 2])
        grid_a = np.asarray(response["amplitude_grid"])
        fresh_state = next(
            s
            for s in response["states"]
            if (s["impedance_factor"], s["background_factor"]) == (1.0, 1.0)
        )
        settled = response["states"][response["settled"]]
        for state, color, label in (
            (fresh_state, WEAK, "fresh electrode"),
            (settled, viz.SERIES[1], "settled electrode"),
        ):
            ax.fill_between(
                grid_a, state["p_keep_p10"], state["p_keep_p90"], color=color, alpha=0.18,
                linewidth=0,
            )  # fmt: skip
            ax.plot(grid_a, state["p_keep"], color=color, marker="o", markersize=3.5, label=label)
        ax.set_xlim(1.02, -0.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Spike amplitude relative to week 0")
        ax.set_ylabel("Share of week-0 crossings still detected")
        ax.legend(loc="upper right", fontsize=8.2)
        viz.panel_label(ax, "Crossings kept (MEArec)")

        at = int(np.argmin(np.abs(weeks - key_week)))
        _headline(
            fig,
            f"Aging model: impedance up {last['z_1khz_kohm'] / first['z_1khz_kohm']:.2f} times "
            f"in two weeks, spikes at {amplitude['p50'][at]:.2f} of their size by week "
            f"{key_week:g}",
            f"Front-end noise of the conventional chain rises from {first['frontend_noise_uv']:.1f}"
            f" to {last['frontend_noise_uv']:.1f} µV (300 Hz to 7.5 kHz, input-referred). With "
            f"spikes unchanged the settled electrode alone keeps {settled['p_keep'][0]:.0%} of "
            "crossings. Bands: 10th to 90th percentile of channels.",
        )
        _provenance(
            fig,
            dataset="Randles electrode with constant-phase double layer (Straka 2018) plus a "
            "series sheath; impedance ratio from Barrese 2013, amplitude decline from Chestek "
            f"2011 (both assumptions); MEArec, {response['n_channels']} channels, 13 µV "
            "background",
            split=f"MEArec: thresholds set on the first 10 s of each condition, "
            f"{response['scored_s_per_channel']:.0f} s per channel scored; line: pooled over "
            "channels, band: 10th to 90th percentile of channels",
            units="|Z| in kΩ; amplitude and crossings kept as a share of week 0; conventional "
            "chain (10 µA, 10 bits, 30 kS/s), threshold 4.5 sigma",
        )
        return viz.save_figure(fig, path)


# ------------------------------------------------------------------------ R² against weeks


def _curve(ax: Axes, rows: list[dict[str, Any]], mode: str, *, band: bool) -> None:
    chosen = sorted(_rows(rows, mode=mode), key=lambda r: r["week"])
    if not chosen:
        return
    weeks = [r["week"] for r in chosen]
    if band:
        ax.fill_between(
            weeks,
            [r["r2_low"] for r in chosen],
            [r["r2_high"] for r in chosen],
            color=MODE_COLORS[mode],
            alpha=0.15,
            linewidth=0,
        )
    ax.plot(
        weeks,
        [r["r2"] for r in chosen],
        color=MODE_COLORS[mode],
        marker="o",
        markersize=3.5,
        linestyle=(0, (4, 3)) if mode == "refit" else "-",
        label=MODE_LABELS[mode],
    )


def plot_weeks(summary: dict[str, Any], path: Path, decoders: tuple[str, ...] = ()) -> Path:
    """Draw R² against simulated weeks for each chain, dataset and decoder."""
    decode, key = summary["decoding"], summary["key"]
    names = list(decode["conditions"])
    available = list(dict.fromkeys(r["decoder"] for r in decode["rows"]))
    shown = [d for d in (decoders or ("wiener", "gru")) if d in available] or available[:1]
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 3.1 * len(shown) + 2.6))
        grid = fig.add_gridspec(
            len(shown), len(names), left=0.065, right=0.985,
            top=1.0 - 1.15 / (3.1 * len(shown) + 2.6), bottom=1.25 / (3.1 * len(shown) + 2.6),
            wspace=0.22, hspace=0.36,
        )  # fmt: skip
        for row, decoder in enumerate(shown):
            for col, condition in enumerate(names):
                ax = fig.add_subplot(grid[row, col])
                rows = _rows(
                    decode["rows"], variation="baseline", condition=condition, decoder=decoder
                )
                for mode in ("fixed", "zscore", "refit"):
                    _curve(ax, rows, mode, band=mode != "refit")
                ax.axvline(key["week"], color=viz.GRID, linewidth=1.0, zorder=0)
                top = max(r["r2_high"] for r in rows)
                ax.set_ylim(min(0.0, min(r["r2"] for r in rows) - 0.03), top + 0.05)
                if row == len(shown) - 1:
                    ax.set_xlabel("Simulated weeks after implantation")
                if col == 0:
                    ax.set_ylabel(f"Held-out R² ({DECODER_LABELS.get(decoder, decoder)})")
                if row == 0 and col == 0:
                    ax.legend(loc="lower left", fontsize=8.0)
                viz.panel_label(ax, _title(condition) if row == 0 else "")

        def at(condition: str, decoder: str, mode: str) -> dict[str, Any]:
            return dict(key["curves"][condition][decoder][f"key_{mode}"])

        cheap, conventional = names[0], names[min(1, len(names) - 1)]
        best = _best_decoder(summary, cheap)
        label = DECODER_LABELS.get(best, best)
        _headline(
            fig,
            f"After {key['week']:g} simulated weeks: 0.48 µW chain R² "
            f"{at(cheap, best, 'fixed')['r2']:.2f} left alone, "
            f"{at(cheap, best, 'zscore')['r2']:.2f} with a running z-score",
            f"{label}; week 0: {key['curves'][cheap][best]['week0']:.2f}. Conventional chain: "
            f"{at(conventional, best, 'fixed')['r2']:.2f} left alone, "
            f"{at(conventional, best, 'zscore')['r2']:.2f} stabilised (week 0: "
            f"{key['curves'][conventional][best]['week0']:.2f}). The stabiliser costs "
            f"{key['stabiliser_uw']:.5f} µW per channel. Vertical line: week {key['week']:g}.",
        )
        info = decode["conditions"]
        _provenance(
            fig,
            dataset="; ".join(
                f"{DATASET_NAMES.get(d['dataset'], d['dataset'])} ({d['n_channels']} channels, "
                f"{d['bin_s'] * 1e3:.0f} ms bins)"
                for d in {i["dataset"]: i for i in info.values()}.values()
            )
            + "; aging measured on MEArec and applied to the recorded features",
            split="decoder fitted at week 0 on 4 of 5 contiguous blocks of trials, tested on "
            "the fifth block aged to each week; bands: 95 % intervals from 1000 resamples of "
            "trials (NoMAD: segments)",
            units="R² of decoded velocity, pooled over held-out bins; weeks are simulated, the "
            "aging parameters are assumptions from the literature",
        )
        return viz.save_figure(fig, path)


# ------------------------------------------------------------------------- the real pair


def _bars(
    ax: Axes,
    rows: list[dict[str, Any]],
    decoders: list[str],
    modes: list[tuple[str, str, str]],
) -> None:
    width = 0.8 / len(modes)
    for slot, (mode, color, label) in enumerate(modes):
        chosen = [_one(rows, decoder=d, mode=mode) for d in decoders]
        x = np.arange(len(decoders)) - 0.4 + width * (slot + 0.5)
        y = np.array([r["r2"] for r in chosen])
        low = np.array([r["r2_low"] for r in chosen])
        high = np.array([r["r2_high"] for r in chosen])
        ax.bar(x, y, width=width * 0.9, color=color, label=label, zorder=2)
        ax.errorbar(
            x, y, yerr=[y - low, high - y], fmt="none", ecolor=viz.INK, elinewidth=1.0, zorder=3
        )
    ax.set_xticks(range(len(decoders)))
    ax.set_xticklabels([DECODER_LABELS.get(d, d).replace("U3's ", "") for d in decoders])
    ax.grid(axis="x", visible=False)
    ax.axhline(0.0, color=viz.INK_SECONDARY, linewidth=0.8)


def plot_cross_session(summary: dict[str, Any], path: Path) -> Path:
    """Draw the real NoMAD session pair next to the simulation at the same gap."""
    cross, decode, key = summary["cross_session"], summary["decoding"], summary["key"]
    decoders = list(dict.fromkeys(r["decoder"] for r in cross["rows"]))
    simulated = next((c for c in decode["conditions"] if c.startswith(cross["train"])), None)
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 6.2))
        grid = fig.add_gridspec(
            1, 2, left=0.065, right=0.985, top=0.76, bottom=0.24, wspace=0.16,
            width_ratios=(1.25, 1.0),
        )  # fmt: skip
        ax = fig.add_subplot(grid[0, 0])
        _bars(
            ax,
            cross["rows"],
            decoders,
            [
                ("fixed", MODE_COLORS["fixed"], MODE_LABELS["fixed"]),
                ("zscore", MODE_COLORS["zscore"], MODE_LABELS["zscore"]),
                ("aligned", MODE_COLORS["aligned"], f"{cross['n_latents']} latents, aligned"),
                ("within", MODE_COLORS["refit"], MODE_LABELS["within"]),
            ],
        )
        ax.set_ylabel("R² on the later session")
        ax.legend(loc="upper left", fontsize=8.0, ncols=2)
        viz.panel_label(ax, f"Measured: two sessions {cross['days_apart']:.1f} days apart")
        top = max(r["r2_high"] for r in cross["rows"]) + 0.14
        ax.set_ylim(min(-0.08, min(r["r2_low"] for r in cross["rows"]) - 0.02), top)

        ax = fig.add_subplot(grid[0, 1])
        if simulated is not None:
            rows = _rows(
                decode["rows"], variation="baseline", condition=simulated, week=key["week"]
            )
            _bars(
                ax,
                rows,
                decoders,
                [
                    ("fixed", MODE_COLORS["fixed"], MODE_LABELS["fixed"]),
                    ("zscore", MODE_COLORS["zscore"], MODE_LABELS["zscore"]),
                    ("refit", MODE_COLORS["refit"], MODE_LABELS["refit"]),
                ],
            )
            ax.set_ylim(min(-0.08, min(r["r2_low"] for r in rows) - 0.02), top)
        viz.panel_label(ax, f"Simulated: the earlier session aged {key['week']:g} weeks")

        best = max(decoders, key=lambda d: _one(cross["rows"], decoder=d, mode="aligned")["r2"])
        fixed = _one(cross["rows"], decoder=best, mode="fixed")
        zscore = _one(cross["rows"], decoder=best, mode="zscore")
        aligned = _one(cross["rows"], decoder=best, mode="aligned")
        within = _one(cross["rows"], decoder=best, mode="within")
        label = DECODER_LABELS.get(best, best)
        _headline(
            fig,
            f"{cross['weeks_apart']:.0f} real weeks apart, a running z-score is not enough; "
            f"aligning {cross['n_latents']} latents recovers most",
            f"{label}: {fixed['r2']:.2f} left alone, {zscore['r2']:.2f} with the z-score, "
            f"{aligned['r2']:.2f} aligned ({aligned['gain']:+.2f} [{aligned['gain_low']:+.2f}, "
            f"{aligned['gain_high']:+.2f}]), {within['r2']:.2f} refitted with labels. In the "
            "simulation the z-score alone nearly closes the gap.",
        )
        shift = cross["shift"]
        fig.text(
            0.012, 0.872,
            f"Between the real sessions {shift['share_rate_up']:.0%} of channels fire more, not "
            f"less ({shift['share_rate_more_than_doubled']:.0%} more than double): electrode "
            "aging alone cannot produce that.",
            ha="left", va="top", fontsize=9.5, color=viz.INK_SECONDARY,
        )  # fmt: skip
        _provenance(
            fig,
            dataset="NoMAD (Karpowicz 2025), participant T11, 192 channels, threshold crossings "
            "in 60 ms bins, cursor velocity; right: aging measured on MEArec applied to "
            f"{cross['train'].replace('nomad_', '')}",
            split=f"left: fitted on all of {cross['train'].replace('nomad_', '')}, tested on all "
            f"of {cross['test'].replace('nomad_', '')}; alignment uses its first "
            f"{cross['calibration_s']:.0f} s without labels; refitted: 5 held-out blocks within "
            "the later session; right: 5 held-out blocks. Whiskers: 95 % intervals from "
            "resampling segments",
            units="R² of decoded cursor velocity, averaged over x and y",
        )
        return viz.save_figure(fig, path)


# ---------------------------------------------------------------------------- robustness


def plot_robustness(summary: dict[str, Any], path: Path, decoder: str = "") -> Path:
    """Draw R² at the key week under each changed assumption, for both chains."""
    decode, key = summary["decoding"], summary["key"]
    names = [c for c in decode["conditions"] if c.startswith("nason")] or list(decode["conditions"])
    decoder = decoder or _best_decoder(summary, names[0])
    variations = ["baseline", *summary["variations"]]
    colors = {names[0]: viz.SERIES[0]} | {n: viz.SERIES[1] for n in names[1:2]}
    with viz.figure_style():
        fig = plt.figure(figsize=(11.0, 6.6))
        grid = fig.add_gridspec(1, 2, left=0.27, right=0.985, top=0.80, bottom=0.21, wspace=0.08)
        y = np.arange(len(variations))[::-1]
        for col, (mode, title) in enumerate(
            (("fixed", "Decoder left alone"), ("zscore", "With the running z-score"))
        ):
            ax = fig.add_subplot(grid[0, col])
            for condition, offset in zip(colors, (0.14, -0.14), strict=False):
                rows = [
                    _one(
                        decode["rows"], variation=v, condition=condition, decoder=decoder,
                        week=key["week"], mode=mode,
                    )
                    for v in variations
                ]  # fmt: skip
                r2 = np.array([r["r2"] for r in rows])
                low = np.array([r["r2_low"] for r in rows])
                high = np.array([r["r2_high"] for r in rows])
                ax.errorbar(
                    r2, y + offset, xerr=[r2 - low, high - r2], fmt="o", markersize=5,
                    color=colors[condition], elinewidth=1.4,
                    label=_title(condition).split(",")[0],
                )  # fmt: skip
                ax.axvline(
                    key["curves"][condition][decoder]["week0"],
                    color=colors[condition],
                    linewidth=1.0,
                    linestyle=(0, (4, 3)),
                )
            ax.axvline(0.0, color=viz.INK_SECONDARY, linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(
                ["Baseline assumptions", *(v[0].upper() + v[1:] for v in variations[1:])]
                if col == 0
                else [""] * len(y)
            )
            ax.set_ylim(-0.6, len(variations) - 0.4)
            ax.set_xlim(-0.5, 0.8)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"Held-out R² at week {key['week']:g}")
            if col == 1:
                ax.legend(loc="lower left", fontsize=8.2)
            viz.panel_label(ax, title)

        def score(variation: str, condition: str, mode: str) -> float:
            return float(
                _one(
                    decode["rows"], variation=variation, condition=condition, decoder=decoder,
                    week=key["week"], mode=mode,
                )["r2"]
            )  # fmt: skip

        gains = [score(v, c, "zscore") - score(v, c, "fixed") for v in variations for c in colors]
        cheap_ahead = sum(
            score(v, names[0], "zscore") >= score(v, names[-1], "zscore") for v in variations
        )
        _headline(
            fig,
            "The stabiliser helps under every changed assumption; how much is left depends "
            "on which",
            f"{DECODER_LABELS.get(decoder, decoder)} at week {key['week']:g}. The running "
            f"z-score changes R² by {min(gains):+.2f} to {max(gains):+.2f}. Stabilised, the "
            f"0.48 µW chain scores at least as high as the conventional chain in {cheap_ahead} of "
            f"{len(variations)} cases. Dashed lines: week 0.",
        )
        info = decode["conditions"][names[0]]
        _provenance(
            fig,
            dataset=f"Nason et al. online data ({info['n_channels']} channels, finger velocity, "
            "50 ms bins); aging measured on MEArec and applied to the recorded features; one "
            "assumption changed per row",
            split="decoder fitted at week 0 on 4 of 5 contiguous blocks of trials, tested on "
            f"the fifth aged to week {key['week']:g}; whiskers: 95 % intervals from 1000 "
            "resamples of trials",
            units="R² of decoded finger velocity, pooled over held-out bins",
        )
        return viz.save_figure(fig, path)


# --------------------------------------------------------------------------- stimulation


def plot_stimulation(summary: dict[str, Any], path: Path) -> Path:
    """Draw whether the stabiliser absorbs U5's stimulation-induced feature shift."""
    stim = summary["stimulation"]
    decoders = list(dict.fromkeys(r["decoder"] for r in stim["rows"]))
    modes = [
        ("fixed", MODE_COLORS["fixed"], "fitted before stimulation, left alone"),
        ("zscore", viz.SERIES[0], "plus running z-score"),
        ("gated zscore", viz.SERIES[1], "plus z-score gated by the stimulator"),
        ("refit", viz.SERIES[2], "refitted with stimulation on"),
    ]
    with viz.figure_style():
        fig = plt.figure(figsize=(8.6, 5.6))
        ax = fig.add_axes((0.09, 0.3, 0.89, 0.47))
        _bars(ax, stim["rows"], decoders, modes)
        for slot, decoder in enumerate(decoders):
            reference = _one(stim["rows"], decoder=decoder, mode="no stimulation")["r2"]
            ax.plot(
                [slot - 0.45, slot + 0.45], [reference, reference], color=viz.INK,
                linewidth=1.0, linestyle=(0, (4, 3)),
                label="no stimulation" if slot == 0 else None,
            )  # fmt: skip
        ax.set_ylabel("R² on stimulated bins")
        ax.set_ylim(0.0, max(r["r2_high"] for r in stim["rows"]) + 0.22)
        ax.legend(loc="upper left", fontsize=8.0, ncols=2)
        best = decoders[-1]
        fixed = _one(stim["rows"], decoder=best, mode="fixed")
        gated = _one(stim["rows"], decoder=best, mode="gated zscore")
        refit = _one(stim["rows"], decoder=best, mode="refit")
        _headline(
            fig,
            "The stabiliser absorbs most of the shift stimulation leaves in SBP",
            f"{DECODER_LABELS.get(best, best)} fitted before stimulation: loss "
            f"{fixed['loss']:+.3f} left alone, {gated['loss']:+.3f} with a gated z-score, "
            f"{refit['loss']:+.3f} refitted.",
        )
        _provenance(
            fig,
            dataset="Nason et al. online data (96 channels, finger velocity, 50 ms bins); "
            f"0.48 µW chain, stimulating electrode on the {stim['scenario']}, "
            f"{stim['treatment']} canceller (U5)",
            split=f"5 contiguous blocks of trials held out in turn; scored on the "
            f"{stim['stimulated_bins']} stimulated bins of {stim['n_bins']}; whiskers: 95 % "
            "intervals from resampling trials",
            units="R² of decoded finger velocity; dashed: the same decoder with no stimulation",
        )
        return viz.save_figure(fig, path)
