"""U4 analysis: three decoders on the U3 configurations, with confidence intervals.

U3 chose a sensing chain using a linear readout. U4 asks whether that choice
survives better decoders. A handful of U3's configurations are decoded again
with U3's ridge readout, the same linear readout properly tuned (a Wiener
filter), a Kalman filter and a small GRU. Every R² comes with a
95 % interval from resampling trials, and the comparison that matters (U3's
choice against the conventional chain) gets a paired interval on the difference.

Two kinds of held-out data are reported:

* **Held-out blocks of trials** within a session, for every dataset.
* **A held-out session:** each decoder is fitted on one NoMAD session and tested
  on the other, 274 days later or earlier, with no recalibration.

The degraded features are the ones U3 used: the same degradation values, read
from ``results/u3_metrics.json``, and the same random streams.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neurochain.analysis import u3
from neurochain.data import nomad
from neurochain.decode.evaluate import (
    Decoder,
    bootstrap_r2,
    bootstrap_r2_difference,
    fit_and_predict,
    held_out_predictions,
)
from neurochain.decode.gru import GruDecoder
from neurochain.decode.kalman import KalmanDecoder
from neurochain.decode.ridge import RidgeDecoder, WienerDecoder
from neurochain.power import ChainConfig, PowerModel
from neurochain.recording import FloatArray

DECODERS = ("ridge", "wiener", "kalman", "gru")

NASON_CHAINS = (
    "sbp, 0.05 µA, 8 b, 2 kS/s",
    "sbp, 0.25 µA, 8 b, 2 kS/s",
    "sbp, 10 µA, 8 b, 2 kS/s",
    "crossings, 1 µA, 8 b, 30 kS/s",
    "crossings, 5 µA, 6 b, 30 kS/s",
    "crossings, 10 µA, 10 b, 30 kS/s",
    "both, 10 µA, 10 b, 30 kS/s",
)
NOMAD_CHAINS = (
    "crossings, 0.5 µA, 8 b, 30 kS/s",
    "crossings, 1 µA, 10 b, 30 kS/s",
    "crossings, 2 µA, 8 b, 30 kS/s",
    "crossings, 10 µA, 10 b, 30 kS/s",
)

Sets = dict[str, tuple[u3.DecodingSet, tuple[str, ...]]]


@dataclass(frozen=True)
class U4Config:
    """Parameters of the U4 analysis.

    Attributes:
        seed: Seed for the bootstrap and the GRU.
        n_folds: Contiguous blocks of trials used as held-out folds.
        n_boot: Bootstrap resamples behind each interval.
        decoders: Which decoders to score.
        ridge_lags: Bins of history given to the ridge readout; ``None`` uses
            each dataset's own setting from U3.
        wiener_history_s: History given to the Wiener filter, in seconds.
        kalman_delays: Delays, in bins, the Kalman filter chooses between.
        kalman_displacement: Whether the Kalman state carries displacement.
        gru_hidden: Hidden units in the GRU.
        gru_epochs: Largest number of training epochs for the GRU.
        example_seconds: Length of the decoded stretch kept for the figure, taken
            from the middle of the first dataset.
    """

    seed: int = 0
    n_folds: int = 5
    n_boot: int = 1000
    decoders: tuple[str, ...] = DECODERS
    ridge_lags: int | None = None
    wiener_history_s: float = 0.5
    kalman_delays: tuple[int, ...] = (0, 1, 2, 3, 4)
    kalman_displacement: bool = True
    gru_hidden: int = 64
    gru_epochs: int = 60
    example_seconds: float = 12.0


def wiener_lags(data: u3.DecodingSet, config: U4Config) -> int:
    """Return the Wiener filter's history in bins for this dataset."""
    return max(1, round(config.wiener_history_s / data.bin_s))


def decoder_factory(name: str, data: u3.DecodingSet, config: U4Config) -> Callable[[], Decoder]:
    """Return a function that builds a fresh decoder of the named kind."""
    if name == "ridge":
        return lambda: RidgeDecoder(n_lags=config.ridge_lags or data.n_lags)
    if name == "wiener":
        return lambda: WienerDecoder(n_lags=wiener_lags(data, config))
    if name == "kalman":
        return lambda: KalmanDecoder(
            lags=config.kalman_delays, integral_state=config.kalman_displacement
        )
    if name == "gru":
        return lambda: GruDecoder(
            hidden=config.gru_hidden, epochs=config.gru_epochs, seed=config.seed
        )
    raise ValueError(f"unknown decoder {name!r}")


def decoder_multiplies(
    name: str, n_features: int, n_targets: int, *, n_lags: int, hidden: int, n_states: int
) -> int:
    """Return the multiplies one decoder needs per time bin.

    * Ridge and Wiener: one weight per feature, lag and output.
    * Steady-state Kalman: ``x = M x + K y``, a ``k x k`` and a ``k x d``
      product.
    * GRU: three gates, each a product with the input and with the hidden
      state, then the linear output layer.
    """
    if name in ("ridge", "wiener"):
        return n_features * n_lags * n_targets
    if name == "kalman":
        return n_states * n_states + n_states * n_features
    if name == "gru":
        return 3 * hidden * (n_features + hidden) + hidden * n_targets
    raise ValueError(f"unknown decoder {name!r}")


def decoder_cost(data: u3.DecodingSet, config: U4Config, model: PowerModel) -> dict[str, Any]:
    """Return each decoder's compute, as multiplies per bin and µW per channel.

    Uses the same energy per multiply as the rest of the power model, as if the
    decoder ran on the implant, and shares it across the channels. Counted for
    one feature per channel.
    """
    n_features, n_targets = data.crossings.shape[1], data.targets.shape[1]
    out = {}
    for name in config.decoders:
        multiplies = decoder_multiplies(
            name,
            n_features,
            n_targets,
            n_lags=wiener_lags(data, config)
            if name == "wiener"
            else (config.ridge_lags or data.n_lags),
            hidden=config.gru_hidden,
            n_states=(2 if config.kalman_displacement else 1) * n_targets,
        )
        watts = multiplies / data.bin_s * model.energy_per_multiply_j
        out[name] = {
            "multiplies_per_bin": multiplies,
            "uw_per_channel": watts / n_features * 1e6,
        }
    return out


def chain_features(
    data: u3.DecodingSet, row: dict[str, Any], position: int, seed: int
) -> FloatArray:
    """Rebuild the features U3 gave this configuration (its first random draw).

    Raises:
        ValueError: If the dataset cannot be scored for this configuration.
    """
    chain = ChainConfig(row["feature"], row["i_total_ua"] * 1e-6, row["bits"], row["fs_hz"])
    degradation = u3.Degradation(row["p_keep"], row["extra_rate_hz"], row["sbp_noise_ratio"])
    features = u3.features_for(data, chain, degradation, np.random.default_rng([seed, position]))
    if features is None:
        raise ValueError(f"{data.name} has no features for {chain.label}")
    return features


def _paired(
    data: u3.DecodingSet, a: FloatArray, b: FloatArray, config: U4Config
) -> dict[str, float]:
    difference, low, high = bootstrap_r2_difference(
        data.targets, a, b, data.groups, n_boot=config.n_boot, seed=config.seed
    )
    return {"difference": difference, "low": low, "high": high}


def _held_out(
    name: str, data: u3.DecodingSet, features: FloatArray, config: U4Config
) -> tuple[FloatArray, list[Decoder]]:
    """Return held-out predictions and the decoder fitted in each fold."""
    make = decoder_factory(name, data, config)
    built: list[Decoder] = []

    def make_and_keep() -> Decoder:
        built.append(make())
        return built[-1]

    predictions = held_out_predictions(
        make_and_keep, features, data.targets, data.groups, n_folds=config.n_folds
    )
    return predictions, built


def evaluate_set(
    data: u3.DecodingSet,
    labels: tuple[str, ...],
    u3_summary: dict[str, Any],
    config: U4Config,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, FloatArray]]:
    """Score every decoder on the named configurations of one dataset.

    Returns:
        ``(rows, headline, example)``: one row per configuration and decoder;
        the comparison of U3's winning configuration with the conventional
        chain under each decoder, and of the decoders with the ridge readout;
        and each decoder's held-out predictions on the winning configuration.
    """
    grid_labels = [chain.label for chain in u3.grid(u3.U3Config())]
    by_label = {row["label"]: row for row in u3_summary["configurations"]}
    seed = u3_summary["config"]["seed"]
    info = u3_summary["datasets"][data.name]
    base = u3_summary["configurations"][info["baseline"]]["label"]
    best = u3_summary["configurations"][info["winner"]]["label"]
    rows: list[dict[str, Any]] = []
    kept: dict[tuple[str, str], FloatArray] = {}
    scores: dict[tuple[str, str], float] = {}
    for label in labels:
        row = by_label[label]
        features = chain_features(data, row, grid_labels.index(label), seed)
        for name in config.decoders:
            predictions, built = _held_out(name, data, features, config)
            r2, low, high = bootstrap_r2(
                data.targets, predictions, data.groups, n_boot=config.n_boot, seed=config.seed
            )
            result = {
                "dataset": data.name,
                "label": label,
                "feature": row["feature"],
                "total_uw": row["total_uw"],
                "decoder": name,
                "r2": r2,
                "r2_low": low,
                "r2_high": high,
            }
            if name == "kalman":
                result["delay_bins"] = [d.lag for d in built if isinstance(d, KalmanDecoder)]
            if name == "wiener":
                result["penalty"] = [d.alpha for d in built if isinstance(d, WienerDecoder)]
            rows.append(result)
            scores[label, name] = r2
            if label in (base, best):
                kept[label, name] = predictions

    headline: dict[str, Any] = {"conventional": base, "winner": best, "decoders": {}}
    if base in labels and best in labels:
        for name in config.decoders:
            entry: dict[str, Any] = {
                "conventional_r2": scores[base, name],
                "winner_r2": scores[best, name],
                "retained": scores[best, name] / scores[base, name],
                "winner_minus_conventional": _paired(
                    data, kept[best, name], kept[base, name], config
                ),
            }
            if "ridge" in config.decoders and name != "ridge":
                entry["winner_minus_ridge"] = _paired(
                    data, kept[best, name], kept[best, "ridge"], config
                )
            headline["decoders"][name] = entry
    example = {name: kept[best, name] for name in config.decoders if (best, name) in kept}
    return rows, headline, example


def cross_session(
    early: u3.DecodingSet, late: u3.DecodingSet, config: U4Config
) -> list[dict[str, Any]]:
    """Fit each decoder on one session and score it on the other, both ways.

    Features are scaled by the training session's mean and spread; nothing is
    refitted on the test session.
    """
    rows = []
    for train, test, direction in ((early, late, "forward"), (late, early, "backward")):
        for name in config.decoders:
            predictions = fit_and_predict(
                decoder_factory(name, train, config),
                (train.crossings, train.targets, train.groups),
                test.crossings,
                test.groups,
            )
            r2, low, high = bootstrap_r2(
                test.targets, predictions, test.groups, n_boot=config.n_boot, seed=config.seed
            )
            rows.append(
                {
                    "train": train.name,
                    "test": test.name,
                    "direction": direction,
                    "decoder": name,
                    "r2": r2,
                    "r2_low": low,
                    "r2_high": high,
                }
            )
    return rows


def default_sets(
    u3_summary: dict[str, Any],
) -> tuple[Sets, tuple[u3.DecodingSet, u3.DecodingSet], float]:
    """Load the Nason online data and both NoMAD sessions.

    Returns:
        ``(sets, session_pair, days_apart)``: each dataset with the
        configurations decoded on it, the earlier and later NoMAD session, and
        the days between them.
    """
    sessions = sorted(nomad.load_sessions(), key=lambda s: s.start_time_ns)
    nomad_sets = [u3.nomad_set(s, u3_summary["config"]["nomad_rebin"]) for s in sessions]
    sets: Sets = {"nason": (u3.nason_set(), NASON_CHAINS)}
    sets.update({d.name: (d, NOMAD_CHAINS) for d in nomad_sets})
    days_apart = abs(nomad.days_between(sessions[0], sessions[1]))
    return sets, (nomad_sets[0], nomad_sets[1]), days_apart


def compute_u4(
    u3_summary: dict[str, Any],
    *,
    config: U4Config | None = None,
    sets: Sets | None = None,
    session_pair: tuple[u3.DecodingSet, u3.DecodingSet] | None = None,
    days_apart: float | None = None,
) -> dict[str, Any]:
    """Run the U4 analysis.

    Args:
        u3_summary: The contents of ``results/u3_metrics.json``.
        config: Analysis parameters.
        sets: Datasets and the configurations to decode on each; defaults to
            the Nason online data and both NoMAD sessions.
        session_pair: The earlier and later session for the held-out-session
            test; defaults to NoMAD's two when ``sets`` is not given, and to no
            test when it is.
        days_apart: Days between the two sessions, for the record.

    Returns:
        The JSON-serialisable results. ``example`` holds a short stretch of true
        behaviour and each decoder's held-out prediction of it on the winning
        configuration, for the trace figure.
    """
    config = config or U4Config()
    if sets is None:
        sets, session_pair, days_apart = default_sets(u3_summary)

    rows: list[dict[str, Any]] = []
    example: dict[str, Any] = {}
    headline: dict[str, Any] = {}
    cost: dict[str, Any] = {}
    model = PowerModel(**u3_summary["config"]["power"])
    for position, (name, (data, labels)) in enumerate(sets.items()):
        set_rows, headline[name], predictions = evaluate_set(data, labels, u3_summary, config)
        rows += set_rows
        cost[name] = decoder_cost(data, config, model)
        if position == 0 and predictions:
            n = round(config.example_seconds / data.bin_s)
            start = max(0, len(data.targets) // 2 - n // 2)
            example = {
                "dataset": name,
                "label": headline[name]["winner"],
                "bin_s": data.bin_s,
                "truth": np.round(data.targets[start : start + n, 0], 5).tolist(),
                **{
                    decoder: np.round(values[start : start + n, 0], 5).tolist()
                    for decoder, values in predictions.items()
                },
            }
    return {
        "config": asdict(config),
        "results": rows,
        "headline": headline,
        "decoder_cost": cost,
        "example": example,
        "cross_session": None
        if session_pair is None
        else {"days_apart": days_apart, "rows": cross_session(*session_pair, config)},
    }
