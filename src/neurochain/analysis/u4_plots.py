"""U4 figures: decoders compared across configurations, example traces, sessions.

Every number drawn comes from the summary written to ``results/u4_metrics.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from neurochain import viz

DECODER_COLORS = {
    "ridge": viz.INK_SECONDARY,
    "wiener": viz.SERIES[0],
    "kalman": viz.SERIES[1],
    "gru": viz.SERIES[2],
}
DECODER_LABELS = {
    "ridge": "U3's ridge readout",
    "wiener": "Wiener filter",
    "kalman": "Kalman filter",
    "gru": "GRU",
}
DATASET_TITLES = {
    "nason": "Nason: finger velocity, 96 channels",
    "nomad_session1": "NoMAD session 1: cursor velocity, 192 channels",
    "nomad_session2": "NoMAD session 2: cursor velocity, 192 channels",
}
MEASURED = "#a3a29c"
TRACE_COLORS = {**DECODER_COLORS, "ridge": viz.INK}
SESSION_NAMES = {"nomad_session1": "session 1", "nomad_session2": "session 2"}
TARGET_LABELS = {"nason": "Finger velocity (full range per second)"}
FEATURE_SHORT = {"crossings": "Crossings", "sbp": "SBP", "both": "Both"}


def _tick(label: str, total_uw: float, role: str) -> str:
    """Turn ``'sbp, 0.25 µA, 8 b, 2 kS/s'`` into a three-line tick label."""
    feature, current, bits, rate = label.split(", ")
    lines = [
        f"{FEATURE_SHORT.get(feature, feature)}, {current}",
        f"{bits}, {rate}",
        f"{total_uw:.2f} µW",
    ]
    if role:
        lines.append(role)
    return "\n".join(lines)


def _decoder_panel(ax: Axes, summary: dict[str, Any], name: str) -> None:
    rows = [r for r in summary["results"] if r["dataset"] == name]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    head = summary["headline"][name]
    roles = {head["conventional"]: "conventional", head["winner"]: "U3's choice"}
    offsets = dict(zip(DECODER_COLORS, (-0.27, -0.09, 0.09, 0.27), strict=True))
    for slot, label in enumerate(labels):
        if label in roles:
            ax.axvspan(slot - 0.45, slot + 0.45, color=viz.NEUTRAL_FILL, alpha=0.35, linewidth=0)
    for decoder, color in DECODER_COLORS.items():
        chosen = {r["label"]: r for r in rows if r["decoder"] == decoder}
        if not chosen:
            continue
        x = np.array([labels.index(label) + offsets[decoder] for label in chosen])
        y = np.array([r["r2"] for r in chosen.values()])
        low = np.array([r["r2_low"] for r in chosen.values()])
        high = np.array([r["r2_high"] for r in chosen.values()])
        ax.errorbar(
            x, y, yerr=[y - low, high - y], fmt="o", color=color, markersize=5.5,
            markeredgecolor=viz.SURFACE, markeredgewidth=1.0, elinewidth=1.6, capsize=0,
            label=DECODER_LABELS[decoder], zorder=3,
        )  # fmt: skip
    power = {r["label"]: r["total_uw"] for r in rows}
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [_tick(label, power[label], roles.get(label, "")) for label in labels], fontsize=7.8
    )
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Held-out R²")
    viz.panel_label(ax, DATASET_TITLES.get(name, name))


def plot_decoders(summary: dict[str, Any], path: Path) -> Path:
    """Draw R² with its interval for every configuration and decoder."""
    names = list(summary["headline"])
    main = names[0]
    head = summary["headline"][main]
    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 9.2))
        grid = fig.add_gridspec(
            2, 2, left=0.075, right=0.985, top=0.87, bottom=0.185, wspace=0.2, hspace=0.62
        )
        ax = fig.add_subplot(grid[0, :] if len(names) > 1 else grid[:, :])
        _decoder_panel(ax, summary, main)
        ax.legend(loc="lower right", ncols=4, fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
        for slot, name in enumerate(names[1:3]):
            _decoder_panel(fig.add_subplot(grid[1, slot]), summary, name)
        kept = [d["retained"] for d in head["decoders"].values()]
        rows = {(r["label"], r["decoder"]): r for r in summary["results"] if r["dataset"] == main}
        win_uw = rows[head["winner"], "ridge"]["total_uw"]
        base_uw = rows[head["conventional"], "ridge"]["total_uw"]
        gain = head["decoders"]["gru"]["winner_minus_ridge"]
        viz.add_headline(
            fig,
            f"The {win_uw:.2f} µW chain keeps {min(kept) * 100:.0f} to {max(kept):.0%} of the "
            f"{base_uw:.1f} µW chain's R² under every decoder",
            f"The decoder moves accuracy more than the front end does: on that chain the GRU adds "
            f"{gain['difference']:.2f} R² over U3's ridge readout. Bars are 95 % intervals.",
        )
        viz.add_provenance(
            fig,
            dataset=(
                "Nason et al. (2020) online data and NoMAD (Karpowicz et al., 2025), with each "
                "configuration's degradation from U3 applied to the recorded features."
            ),
            split=(
                "every bin predicted by a decoder fitted on the other four contiguous blocks of "
                "trials; intervals from 1000 resamples of trials (NoMAD: of recording segments)."
            ),
            units="R² is unitless; µW per channel from the U3 power model. Shaded: the conventional chain and U3's choice.",
        )
        return viz.save_figure(fig, path)


def plot_traces(summary: dict[str, Any], path: Path) -> Path:
    """Draw a stretch of true behaviour against each decoder's held-out prediction."""
    example = summary["example"]
    name = example["dataset"]
    truth = np.asarray(example["truth"])
    time_s = np.arange(truth.size) * example["bin_s"]
    scores = {
        r["decoder"]: r
        for r in summary["results"]
        if r["dataset"] == name and r["label"] == example["label"]
    }
    decoders = [d for d in DECODER_COLORS if d in example]
    with viz.figure_style():
        fig, axes = plt.subplots(
            len(decoders), 1, figsize=(10.0, 1.9 * len(decoders) + 2.3), sharex=True, sharey=True,
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.885, "bottom": 0.16, "hspace": 0.38},
        )  # fmt: skip
        for ax, decoder in zip(np.atleast_1d(axes), decoders, strict=True):
            ax.plot(time_s, truth, color=MEASURED, linewidth=1.4, label="Measured")
            ax.plot(time_s, example[decoder], color=TRACE_COLORS[decoder], linewidth=1.8,
                    label="Decoded")  # fmt: skip
            row = scores[decoder]
            viz.panel_label(
                ax,
                f"{DECODER_LABELS[decoder]}: R² {row['r2']:.2f} "
                f"[{row['r2_low']:.2f}, {row['r2_high']:.2f}] over the whole session",
            )
            ax.legend(loc="upper right", ncols=2, fontsize=8.5)
        np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
        np.atleast_1d(axes)[len(decoders) // 2].set_ylabel(TARGET_LABELS.get(name, "Velocity"))
        viz.add_headline(
            fig,
            "The GRU follows the fast movements that the linear decoders smooth over",
            f"{truth.size * example['bin_s']:.0f} s from the middle of the session, decoded from "
            f"the {scores['ridge']['total_uw']:.2f} µW chain ({example['label'].replace('sbp', 'SBP')}).",
        )
        viz.add_provenance(
            fig,
            dataset="Nason et al. (2020) online data, 96 channels, 50 ms bins, with the chain's degradation from U3 applied.",
            split="each trace is predicted by a decoder fitted on the other four contiguous blocks of trials.",
            units="finger velocity as a fraction of the full flexion range per second; time in seconds.",
        )
        return viz.save_figure(fig, path)


def plot_cross_session(summary: dict[str, Any], path: Path) -> Path:
    """Draw each decoder's R² within a session against the same decoder moved across sessions."""
    cross = summary["cross_session"]
    tests = list(dict.fromkeys(r["test"] for r in cross["rows"]))
    with viz.figure_style():
        fig, axes = plt.subplots(
            1, len(tests), figsize=(10.0, 6.2), sharey=True,
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.83, "bottom": 0.22, "wspace": 0.08},
        )  # fmt: skip
        best_across = max(r["r2"] for r in cross["rows"])
        for ax, test in zip(np.atleast_1d(axes), tests, strict=True):
            base = summary["headline"][test]["conventional"]
            within = {
                r["decoder"]: r
                for r in summary["results"]
                if r["dataset"] == test and r["label"] == base
            }
            across = {r["decoder"]: r for r in cross["rows"] if r["test"] == test}
            train = next(iter(across.values()))["train"]
            shown = [d for d in DECODER_COLORS if d in across]
            for slot, decoder in enumerate(shown):
                color = DECODER_COLORS[decoder]
                for row, shift, filled in (
                    (within[decoder], -0.14, True),
                    (across[decoder], 0.14, False),
                ):
                    ax.errorbar(
                        [slot + shift], [row["r2"]],
                        yerr=[[row["r2"] - row["r2_low"]], [row["r2_high"] - row["r2"]]],
                        fmt="o", color=color, markersize=7.5, elinewidth=1.6,
                        markerfacecolor=color if filled else viz.SURFACE, markeredgecolor=color if not filled else viz.SURFACE,
                        markeredgewidth=1.6 if not filled else 1.0, zorder=3,
                    )  # fmt: skip
                    ax.annotate(f"{row['r2']:.2f}", (slot + shift, row["r2_high"]), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=8.5)  # fmt: skip
            ax.axhline(0, color=viz.INK_SECONDARY, linewidth=0.8)
            ax.set_xticks(range(len(shown)))
            ax.set_xticklabels([DECODER_LABELS[d].replace(" ", "\n", 1) for d in shown])
            ax.set_xlim(-0.6, len(shown) - 0.4)
            ax.grid(axis="x", visible=False)
            viz.panel_label(
                ax,
                f"Tested on {SESSION_NAMES.get(test, test)} (open: fitted on "
                f"{SESSION_NAMES.get(train, train)})",
            )
        np.atleast_1d(axes)[0].set_ylabel("R² on held-out data")
        viz.add_headline(
            fig,
            f"Moved {cross['days_apart']:.0f} days to another session, no decoder scores above R² {best_across:.2f}",
            "A better decoder does not fix a changed recording: the map from channels to movement has moved. U6 tests a stabiliser.",
        )
        viz.add_provenance(
            fig,
            dataset="NoMAD (Karpowicz et al., 2025), one participant, two sessions, 192 channels, crossings in 60 ms bins.",
            split=(
                "filled, fitted in the same session and scored on held-out blocks (5 folds); open, fitted on "
                "the other session, no recalibration."
            ),
            units="R² is unitless, averaged over the two cursor axes; bars are 95 % intervals from resampling recording segments.",
        )
        return viz.save_figure(fig, path)
