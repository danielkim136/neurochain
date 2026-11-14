"""U3 figures: the accuracy-against-power Pareto, power breakdown and the bridge.

Every number drawn comes from the summary written to ``results/u3_metrics.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from neurochain import viz

FEATURE_COLORS = {"crossings": viz.SERIES[0], "sbp": viz.SERIES[1], "both": viz.SERIES[2]}
FEATURE_LABELS = {
    "crossings": "Threshold crossings",
    "sbp": "Spiking-band power",
    "both": "Both, from one wideband chain",
}
DATASET_TITLES = {
    "nason": "Nason: finger velocity, 96 channels",
    "nomad_session1": "NoMAD session 1: cursor velocity",
    "nomad_session2": "NoMAD session 2: cursor velocity",
}
FEATURE_SHORT = {"crossings": "crossings", "sbp": "SBP", "both": "both features", "raw": "raw"}
STAGES = (
    ("lna_uw", "Amplifier"),
    ("adc_uw", "ADC"),
    ("dsp_uw", "Digital logic"),
    ("link_uw", "Radio"),
)


def _plain(value: float, _: object) -> str:
    return f"{value:g}"


def _short(row: dict[str, Any]) -> str:
    """Describe a configuration in a few words."""
    return f"{row['i_total_ua']:g} µA, {row['bits']} b, {row['fs_hz'] / 1e3:g} kS/s"


def _pareto_panel(ax: Axes, summary: dict[str, Any], name: str, *, annotate: bool) -> None:
    rows = summary["configurations"]
    info = summary["datasets"][name]
    for feature, color in FEATURE_COLORS.items():
        chosen = [r for r in rows if r["feature"] == feature and name in r["r2"]]
        if not chosen:
            continue
        ax.scatter(
            [r["total_uw"] for r in chosen], [r["r2"][name]["mean"] for r in chosen], s=22,
            color=color, alpha=0.75, edgecolor=viz.SURFACE, linewidth=0.5,
            label=FEATURE_LABELS[feature], zorder=3,
        )  # fmt: skip
    front = [rows[i] for i in info["pareto"]]
    ax.step(
        [r["total_uw"] for r in front], [r["r2"][name]["mean"] for r in front], where="post",
        color=viz.INK, linewidth=1.2, zorder=2, label="Pareto front",
    )  # fmt: skip
    raw = next((r for r in rows if r["feature"] == "raw" and name in r["r2"]), None)
    if raw is not None:
        ax.scatter([raw["total_uw"]], [raw["r2"][name]["mean"]], s=46, marker="D", color=viz.INK,
                   edgecolor=viz.SURFACE, linewidth=0.8, zorder=4, label="Raw stream, decoded off-implant")  # fmt: skip
    marks = (
        ("baseline", "Conventional chain", (0.97, 0.36), "right"),
        ("winner", "Cheapest match", (0.03, 0.97), "left"),
    )
    for key, text, anchor, align in marks:
        row = rows[info[key]]
        x, y = row["total_uw"], row["r2"][name]["mean"]
        ax.scatter([x], [y], s=150, facecolor="none", edgecolor=viz.INK, linewidth=1.4, zorder=5)
        if annotate:
            ax.annotate(
                f"{text}: {FEATURE_SHORT[row['feature']]}, {_short(row)}\n"
                f"{x:.2f} µW per channel, R² {y:.2f}",
                xy=(x, y), xytext=anchor, textcoords="axes fraction", fontsize=8.5, ha=align,
                va="top",
                arrowprops={"arrowstyle": "-", "color": viz.INK_SECONDARY, "linewidth": 0.8,
                            "shrinkB": 7},
            )  # fmt: skip
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Held-out R²")
    viz.panel_label(ax, DATASET_TITLES.get(name, name))


def plot_pareto(summary: dict[str, Any], path: Path) -> Path:
    """Draw accuracy against power per channel for every configuration."""
    rows = summary["configurations"]
    names = list(summary["datasets"])
    main = names[0]
    info = summary["datasets"][main]
    best, base = rows[info["winner"]], rows[info["baseline"]]
    tolerance = summary["config"]["tolerance"]
    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 8.6))
        grid = fig.add_gridspec(
            2, 2, left=0.085, right=0.975, top=0.87, bottom=0.165, wspace=0.24, hspace=0.42,
            width_ratios=[1.35, 1.0],
        )  # fmt: skip
        ax = fig.add_subplot(grid[:, 0] if len(names) > 1 else grid[:, :])
        _pareto_panel(ax, summary, main, annotate=True)
        ax.set_xlabel("Power per channel (µW)")
        ax.legend(loc="lower right", fontsize=8.5)
        for slot, name in enumerate(names[1:3]):
            side = fig.add_subplot(grid[slot, 1])
            _pareto_panel(side, summary, name, annotate=False)
            if slot == len(names[1:3]) - 1:
                side.set_xlabel("Power per channel (µW)")
        saving = base["total_uw"] / best["total_uw"]
        viz.add_headline(
            fig,
            f"{best['total_uw']:.2f} µW per channel decodes within {tolerance:.0%} of a "
            f"{base['total_uw']:.1f} µW conventional chain",
            f"{saving:.0f} times less power, by sending {FEATURE_LABELS[best['feature']].lower()} instead of "
            "counting spikes. Each point is one configuration.",
        )
        cross = summary["cross_session"]
        drift = (
            ""
            if cross is None
            else f" Across the two NoMAD sessions, {cross['days_apart']:.0f} days apart, a fixed decoder "
            f"scores R² {cross['train_early_test_late']:.2f} and {cross['train_late_test_early']:.2f}."
        )
        viz.add_provenance(
            fig,
            dataset=(
                "Nason et al. (2020) online data and NoMAD (Karpowicz et al., 2025); each "
                "configuration's effect on the features is measured on a MEArec simulation."
            ),
            split="5-fold cross-validation over contiguous blocks of trials within a session; linear readout.",
            units="µW per channel from the power model; R² is unitless." + drift,
        )
        return viz.save_figure(fig, path)


def plot_power(summary: dict[str, Any], path: Path) -> Path:
    """Draw where the power goes in a few named configurations."""
    rows = summary["configurations"]
    names = list(summary["datasets"])
    main = summary["datasets"][names[0]]
    picks: list[tuple[str, dict[str, Any]]] = []
    raw = next((r for r in rows if r["feature"] == "raw"), None)
    if raw is not None:
        picks.append(("Stream every sample", raw))
    picks.append(("Conventional: crossings", rows[main["baseline"]]))
    picks.append(("Most accurate", rows[main["most_accurate"]]))
    for name in names[1:]:
        picks.append((f"Cheapest match, {name.replace('nomad_', 'NoMAD ').replace('session', 'session ')}",
                      rows[summary["datasets"][name]["winner"]]))  # fmt: skip
    picks.append((f"Cheapest match, {names[0].capitalize()}", rows[main["winner"]]))
    win = rows[main["winner"]]
    with viz.figure_style():
        fig, ax = plt.subplots(
            figsize=(10.0, 5.6),
            gridspec_kw={"left": 0.30, "right": 0.87, "top": 0.80, "bottom": 0.25},
        )
        y = np.arange(len(picks))[::-1]
        left = np.zeros(len(picks))
        for (key, label), color in zip(STAGES, viz.SERIES_STACK, strict=True):
            share = np.array([row[key] / row["total_uw"] for _, row in picks])
            ax.barh(y, share, left=left, height=0.62, color=color, edgecolor=viz.SURFACE,
                    linewidth=1.2, label=label)  # fmt: skip
            left += share
        for position, (_, row) in zip(y, picks, strict=True):
            ax.text(1.015, position, f"{row['total_uw']:.2f} µW", va="center", fontsize=9)
        ax.set_yticks(
            y, [f"{title}\n{FEATURE_SHORT[row['feature']]}, {_short(row)}" for title, row in picks]
        )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Share of that configuration's power")
        ax.grid(visible=False, axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=4, fontsize=8.5)
        viz.add_headline(
            fig,
            f"The amplifier is {win['lna_uw'] / win['total_uw']:.0%} of the cheapest chain's "
            f"{win['total_uw']:.2f} µW per channel",
            "Share of per-channel power by stage, total at the right. The radio only matters when every sample is sent.",
        )
        power = summary["config"]["power"]
        viz.add_provenance(
            fig,
            dataset="none; this is the power model applied to configurations from the sweep.",
            split="not applicable.",
            units=(
                f"µW per channel. Assumed: {power['v_supply_v']:g} V, NEF {power['nef']:g}, ADC "
                f"{power['adc_fom_j'] * 1e15:g} fJ/step, {power['energy_per_multiply_j'] * 1e12:g} pJ per "
                f"multiply, {power['energy_per_bit_j'] * 1e12:g} pJ per bit."
            ),
        )
        return viz.save_figure(fig, path)


def plot_bridge(summary: dict[str, Any], path: Path) -> Path:
    """Draw how features degrade with supply current, and the bridge's own check."""
    rows = summary["configurations"]
    name = next(iter(summary["datasets"]))
    quality = summary["crossing_quality"]
    check = summary["bridge_check"]
    settings = ((10, 30_000.0), (6, 30_000.0), (8, 10_000.0))
    with viz.figure_style():
        fig, (ax, ax_r2) = plt.subplots(
            1, 2, figsize=(10.0, 6.4),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.83, "bottom": 0.215, "wspace": 0.24},
        )  # fmt: skip
        for (bits, fs), color in zip(settings, viz.SERIES, strict=True):
            chosen = sorted((q for q in quality if q["bits"] == bits and q["fs_hz"] == fs),
                            key=lambda q: q["i_total_ua"])  # fmt: skip
            ax.plot([q["i_total_ua"] for q in chosen], [q["p_keep"] for q in chosen], color=color,
                    linewidth=2.0, marker="o", markersize=5, markeredgecolor=viz.SURFACE,
                    markeredgewidth=1.0, label=f"{bits} bits, {fs / 1e3:g} kS/s")  # fmt: skip
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Amplifier supply current (µA)")
        ax.set_ylabel("Share of the reference chain's real spikes still detected")
        ax.legend(loc="lower right", fontsize=8.5)
        viz.panel_label(ax, "Crossings: what a cheaper chain keeps (MEArec)")

        for feature, color in FEATURE_COLORS.items():
            chosen = sorted(
                (r for r in rows if r["feature"] == feature and r["bits"] == 8
                 and r["fs_hz"] in (2000.0, 30_000.0) and name in r["r2"]),
                key=lambda r: r["i_total_ua"],
            )  # fmt: skip
            ax_r2.plot([r["i_total_ua"] for r in chosen], [r["r2"][name]["mean"] for r in chosen],
                       color=color, linewidth=2.0, marker="o", markersize=5,
                       markeredgecolor=viz.SURFACE, markeredgewidth=1.0,
                       label=FEATURE_LABELS[feature])  # fmt: skip
        ax_r2.set_xscale("log")
        ax_r2.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax_r2.set_ylim(bottom=0)
        ax_r2.set_xlabel("Amplifier supply current (µA)")
        ax_r2.set_ylabel("Held-out R²")
        ax_r2.legend(loc="lower right", fontsize=8.5)
        viz.panel_label(
            ax_r2, f"{DATASET_TITLES.get(name, name).split(':')[0]}: decoding at 8 bits"
        )
        viz.add_headline(
            fig,
            "Spiking-band power holds its accuracy to a tenth of the current that crossings need",
            f"Bridge check at {check['low_current_ua']:g} µA: crossings follow the true spike counts at r = "
            f"{check['crossings_r_actual']:.2f}, the bridge predicts {check['crossings_r_predicted']:.2f}; "
            f"SBP {check['sbp_r_actual']:.3f} against {check['sbp_r_predicted']:.3f}.",
        )
        viz.add_provenance(
            fig,
            dataset=(
                "left, MEArec simulation at 13 µV background noise, all 32 channels; right, "
                "Nason et al. (2020) online data degraded by the bridge."
            ),
            split="left, thresholds set on the first 10 s and compared on the rest; right, 5-fold blocked cross-validation.",
            units="shares and R² are unitless. The wideband amplifier cannot reach 7.5 kHz below 0.25 µA, so its curves stop there.",
        )
        return viz.save_figure(fig, path)
