import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis import u3, u4
from neurochain.analysis.u4_plots import plot_cross_session, plot_decoders, plot_traces
from neurochain.power import ChainConfig, PowerModel, chain_power

CONVENTIONAL = ChainConfig("crossings", 10e-6, 10, 30_000.0)
CHEAP = ChainConfig("sbp", 0.25e-6, 8, 2000.0)
SMALL = u4.U4Config(n_folds=3, n_boot=200, gru_hidden=8, gru_epochs=6, example_seconds=5.0)


def decoding_set(name: str, seed: int = 0, *, with_sbp: bool = True) -> u3.DecodingSet:
    """Twelve channels whose rates follow a smooth hidden two-dimensional velocity."""
    rng = np.random.default_rng(seed)
    n = 2400
    velocity = np.stack(
        [np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode="same") * 3 for _ in range(2)],
        axis=1,
    )
    rate = np.exp(1.0 + 0.5 * velocity @ rng.standard_normal((2, 12)))
    return u3.DecodingSet(
        name=name,
        crossings=rng.poisson(rate).astype(float),
        sbp=10.0 + 2.0 * rate / rate.mean() + 0.3 * rng.standard_normal(rate.shape)
        if with_sbp
        else None,
        targets=velocity,
        groups=np.repeat(np.arange(n // 40), 40),
        bin_s=0.05,
        n_lags=3,
    )


def fake_u3_summary(*names: str) -> dict:
    """The parts of ``results/u3_metrics.json`` that U4 reads."""
    rows = [
        {
            "label": chain.label,
            "feature": chain.feature,
            "i_total_ua": chain.i_total_a * 1e6,
            "bits": chain.bits,
            "fs_hz": chain.fs_hz,
            "total_uw": chain_power(chain).total_w * 1e6,
            "p_keep": 1.0,
            "extra_rate_hz": 0.0,
            "sbp_noise_ratio": ratio,
        }
        for chain, ratio in ((CONVENTIONAL, 1.0), (CHEAP, 1.5))
    ]
    return {
        "config": {"seed": 0, "nomad_rebin": 3, "power": asdict(PowerModel())},
        "configurations": rows,
        "datasets": {name: {"baseline": 0, "winner": 1} for name in names},
    }


@pytest.fixture(scope="module")
def summary() -> dict:
    toy, other = decoding_set("toy"), decoding_set("other", seed=1)
    return u4.compute_u4(
        fake_u3_summary("toy", "other"),
        config=SMALL,
        sets={
            "toy": (toy, (CHEAP.label, CONVENTIONAL.label)),
            "other": (other, (CHEAP.label, CONVENTIONAL.label)),
        },
        session_pair=(toy, other),
        days_apart=10.0,
    )


def test_decoder_multiplies_for_known_sizes() -> None:
    sizes = {"n_lags": 5, "hidden": 64, "n_states": 2}
    assert u4.decoder_multiplies("ridge", 96, 1, **sizes) == 480
    assert u4.decoder_multiplies("wiener", 96, 1, **sizes) == 480
    assert u4.decoder_multiplies("kalman", 96, 1, **sizes) == 196
    assert u4.decoder_multiplies("gru", 96, 1, **sizes) == 30_784
    with pytest.raises(ValueError, match="unknown decoder"):
        u4.decoder_multiplies("lstm", 96, 1, **sizes)
    with pytest.raises(ValueError, match="unknown decoder"):
        u4.decoder_factory("lstm", decoding_set("toy"), SMALL)


def test_every_configuration_and_decoder_has_a_score_with_an_interval(summary: dict) -> None:
    rows = summary["results"]
    assert len(rows) == 2 * 2 * 4
    assert {r["decoder"] for r in rows} == set(u4.DECODERS)
    for row in rows:
        assert row["r2_low"] < row["r2"] < row["r2_high"]
        assert row["total_uw"] > 0
    linear = [r for r in rows if r["decoder"] != "gru"]
    wiener = [r for r in rows if r["decoder"] == "wiener"]
    assert all(len(r["penalty"]) == SMALL.n_folds for r in wiener)
    assert min(r["r2"] for r in linear) > 0.3
    kalman = [r for r in rows if r["decoder"] == "kalman"]
    assert all(len(r["delay_bins"]) == SMALL.n_folds for r in kalman)


def test_headline_compares_the_two_chains_with_paired_intervals(summary: dict) -> None:
    head = summary["headline"]["toy"]
    assert head["conventional"] == CONVENTIONAL.label
    assert head["winner"] == CHEAP.label
    rows = {
        (r["label"], r["decoder"]): r["r2"] for r in summary["results"] if r["dataset"] == "toy"
    }
    for decoder, entry in head["decoders"].items():
        assert entry["winner_r2"] == rows[CHEAP.label, decoder]
        assert entry["conventional_r2"] == rows[CONVENTIONAL.label, decoder]
        diff = entry["winner_minus_conventional"]
        assert diff["low"] <= diff["difference"] <= diff["high"]
        assert diff["difference"] == pytest.approx(entry["winner_r2"] - entry["conventional_r2"])
        assert ("winner_minus_ridge" in entry) == (decoder != "ridge")


def test_decoder_cost_ranks_the_gru_most_expensive(summary: dict) -> None:
    cost = summary["decoder_cost"]["toy"]
    assert cost["gru"]["multiplies_per_bin"] > cost["ridge"]["multiplies_per_bin"]
    assert cost["ridge"]["multiplies_per_bin"] == 12 * 3 * 2
    assert cost["wiener"]["multiplies_per_bin"] == 12 * 10 * 2  # 0.5 s of 50 ms bins
    expected_uw = cost["ridge"]["multiplies_per_bin"] / 0.05 * 1e-12 / 12 * 1e6
    assert cost["ridge"]["uw_per_channel"] == pytest.approx(expected_uw)


def test_cross_session_rows_and_example(summary: dict) -> None:
    cross = summary["cross_session"]
    assert cross["days_apart"] == 10.0
    assert len(cross["rows"]) == 8
    assert {(r["train"], r["test"]) for r in cross["rows"]} == {("toy", "other"), ("other", "toy")}
    # The two toy sessions have unrelated tuning, so a moved decoder fails.
    within = min(r["r2"] for r in summary["results"] if r["decoder"] == "ridge")
    assert max(r["r2"] for r in cross["rows"] if r["decoder"] == "ridge") < within - 0.2
    example = summary["example"]
    assert example["dataset"] == "toy"
    assert example["label"] == CHEAP.label
    assert all(len(example[key]) == 100 for key in ("truth", *u4.DECODERS))
    json.dumps(summary)


def test_changed_settings_reach_the_decoders() -> None:
    toy = decoding_set("toy")
    config = u4.U4Config(
        n_folds=3,
        n_boot=50,
        decoders=("ridge", "kalman"),
        ridge_lags=1,
        kalman_delays=(2,),
        kalman_displacement=False,
    )
    result = u4.compute_u4(
        fake_u3_summary("toy"), config=config, sets={"toy": (toy, (CONVENTIONAL.label,))}
    )
    assert {r["decoder"] for r in result["results"]} == {"ridge", "kalman"}
    kalman = next(r for r in result["results"] if r["decoder"] == "kalman")
    assert kalman["delay_bins"] == [2, 2, 2]
    assert result["decoder_cost"]["toy"]["ridge"]["multiplies_per_bin"] == 12 * 1 * 2
    assert result["decoder_cost"]["toy"]["kalman"]["multiplies_per_bin"] == 2 * 2 + 2 * 12
    assert result["cross_session"] is None
    assert result["headline"]["toy"]["decoders"] == {}  # only one of the two chains was decoded


def test_a_dataset_without_sbp_cannot_decode_an_sbp_chain() -> None:
    counts_only = decoding_set("toy", with_sbp=False)
    row = fake_u3_summary("toy")["configurations"][1]
    with pytest.raises(ValueError, match="no features"):
        u4.chain_features(counts_only, row, 0, 0)


def test_figures_are_written(summary: dict, tmp_path: Path) -> None:
    for plot in (plot_decoders, plot_traces, plot_cross_session):
        path = plot(summary, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000
