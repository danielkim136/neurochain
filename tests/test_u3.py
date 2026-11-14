import json
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis import u2, u3
from neurochain.analysis.u3_plots import plot_bridge, plot_pareto, plot_power
from neurochain.data.mearec import synthetic_recording
from neurochain.frontend.model import BandwidthError, design_lna
from neurochain.power import ChainConfig, PowerModel

DETECTION = u2.U2Config(calibration_s=2.0, alias_zones=2, max_workers=1)
SMALL = u3.U3Config(
    currents_a=(0.025e-6, 0.5e-6, 10e-6),
    crossing_bits=(8, 10),
    crossing_fs_hz=(30_000.0,),
    sbp_bits=(8,),
    reference_noise_uv=13.0,
    n_repeats=1,
)


def decoding_set(name: str = "toy", *, with_sbp: bool = True) -> u3.DecodingSet:
    """Twelve channels whose rates follow a smooth hidden velocity."""
    rng = np.random.default_rng(0)
    n = 4000
    velocity = np.convolve(rng.standard_normal(n), np.ones(20) / 20, mode="same") * 4
    tuning = rng.standard_normal(12)
    rate = np.exp(1.0 + 0.6 * velocity[:, None] * tuning)
    crossings = rng.poisson(rate).astype(float)
    sbp = 10.0 + 2.0 * rate / rate.mean() + 0.3 * rng.standard_normal(rate.shape)
    return u3.DecodingSet(
        name=name,
        crossings=crossings,
        sbp=sbp if with_sbp else None,
        targets=velocity[:, None],
        groups=np.repeat(np.arange(n // 50), 50),
        bin_s=0.05,
        n_lags=3,
    )


@pytest.fixture(scope="module")
def summary() -> dict:
    return u3.compute_u3(
        synthetic_recording(seed=1),
        config=SMALL,
        detection=DETECTION,
        sets=[decoding_set("toy"), decoding_set("counts_only", with_sbp=False)],
    )


def test_amplifier_has_a_current_floor_for_its_bandwidth() -> None:
    with pytest.raises(BandwidthError, match="reaches at most"):
        design_lna(i_total_a=0.025e-6)
    # The narrow band is reachable at the same current.
    assert design_lna(f_high_hz=1000.0, f_low_hz=300.0, i_total_a=0.025e-6).c_load_f > 0


def test_grid_and_infeasible_configurations(summary: dict) -> None:
    labels = {row["label"] for row in summary["configurations"]}
    assert "raw, 10 µA, 10 b, 30 kS/s" in labels
    assert "sbp, 0.025 µA, 8 b, 2 kS/s" in labels  # the narrow chain works at 25 nA
    assert not any(label.startswith("crossings, 0.025") for label in labels)
    reasons = {item["label"]: item["reason"] for item in summary["infeasible"]}
    assert "crossings, 0.025 µA, 8 b, 30 kS/s" in reasons
    assert "7500 Hz" in reasons["crossings, 0.025 µA, 8 b, 30 kS/s"]
    json.dumps(summary)


def test_reference_chain_leaves_the_recording_untouched(summary: dict) -> None:
    row = next(r for r in summary["configurations"] if r["label"] == u3.REFERENCE.label)
    assert (row["p_keep"], row["extra_rate_hz"], row["sbp_noise_ratio"]) == (1.0, 0.0, 1.0)
    assert row["r2"]["toy"]["n"] == 1
    assert row["total_uw"] == pytest.approx(19.627, abs=1e-3)


def test_less_current_degrades_each_feature(summary: dict) -> None:
    rows = {row["label"]: row for row in summary["configurations"]}
    low, high = rows["crossings, 0.5 µA, 10 b, 30 kS/s"], rows["crossings, 10 µA, 10 b, 30 kS/s"]
    assert low["p_keep"] < high["p_keep"] == 1.0
    assert low["r2"]["toy"]["mean"] < high["r2"]["toy"]["mean"]
    tiny, ample = rows["sbp, 0.025 µA, 8 b, 2 kS/s"], rows["sbp, 10 µA, 8 b, 2 kS/s"]
    assert tiny["sbp_noise_ratio"] > ample["sbp_noise_ratio"] >= 1.0
    assert tiny["r2"]["toy"]["mean"] < ample["r2"]["toy"]["mean"]
    assert tiny["total_uw"] < 0.1 < low["total_uw"]


def test_datasets_without_sbp_only_score_crossings(summary: dict) -> None:
    for row in summary["configurations"]:
        scored = "counts_only" in row["r2"]
        assert scored == (row["feature"] in ("crossings", "raw"))


def test_pareto_and_winner(summary: dict) -> None:
    rows = summary["configurations"]
    for name, info in summary["datasets"].items():
        front = [rows[i] for i in info["pareto"]]
        power = [r["total_uw"] for r in front]
        accuracy = [r["r2"][name]["mean"] for r in front]
        assert power == sorted(power)
        assert accuracy == sorted(accuracy)
        best, base = rows[info["winner"]], rows[info["baseline"]]
        assert base["label"] == u3.REFERENCE.label
        assert best["total_uw"] <= base["total_uw"]
        assert best["r2"][name]["mean"] >= 0.95 * info["baseline_r2"]
        assert rows[info["most_accurate"]]["r2"][name]["mean"] >= info["baseline_r2"]


def test_pareto_front_and_winner_on_known_points() -> None:
    power = np.array([1.0, 2.0, 3.0, 4.0, 0.5])
    accuracy = np.array([0.5, 0.4, 0.7, 0.7, 0.1])
    assert u3.pareto_front(power, accuracy).tolist() == [4, 0, 2]
    assert u3.winner(power, accuracy, 0.45) == 0
    assert u3.winner(power, accuracy, 0.65) == 2


def test_bridge_check_is_reported(summary: dict) -> None:
    check = summary["bridge_check"]
    assert check["low_current_ua"] == 0.5
    assert 0.0 < check["p_keep"] <= 1.0
    for feature in ("crossings", "sbp"):
        for stat in ("mean", "sd", "r"):
            assert f"{feature}_{stat}_actual" in check and f"{feature}_{stat}_predicted" in check
    # On data with truth the bridge's SBP agrees with the directly simulated one.
    assert check["sbp_mean_predicted"] == pytest.approx(check["sbp_mean_actual"], rel=0.15)
    assert check["sbp_r_predicted"] == pytest.approx(check["sbp_r_actual"], abs=0.1)


def test_rescore_moves_power_not_accuracy(summary: dict) -> None:
    base = u3.rescore(summary, PowerModel(), 0.05)
    assert base["toy"]["baseline_uw"] == pytest.approx(19.627, abs=1e-3)
    assert base["raw_stream"]["total_uw"] == pytest.approx(22.536, abs=1e-3)
    efficient = u3.rescore(summary, PowerModel(nef=2.0), 0.05)
    # Half the NEF is a quarter of the amplifier current for the same noise.
    assert efficient["toy"]["baseline_uw"] == pytest.approx(18.0 / 4 + 1.627, abs=1e-2)
    assert efficient["toy"]["baseline_r2"] == base["toy"]["baseline_r2"]
    costly_radio = u3.rescore(summary, PowerModel(energy_per_bit_j=1e-9), 0.05)
    assert costly_radio["raw_stream"]["total_uw"] > 300.0
    loose = u3.rescore(summary, PowerModel(), 0.5)
    assert loose["toy"]["winner_uw"] <= base["toy"]["winner_uw"]


def test_cross_session_score_and_features_for() -> None:
    a, b = decoding_set("a"), decoding_set("b")
    assert u3.cross_session_r2(a, b) > 0.3  # same tuning, so a fixed decoder transfers
    rng = np.random.default_rng(0)
    both = u3.features_for(a, ChainConfig("both", 1e-6), u3.Degradation(), rng)
    assert both is not None and both.shape == (4000, 24)
    counts_only = decoding_set("c", with_sbp=False)
    assert u3.features_for(counts_only, ChainConfig("sbp", 1e-6), u3.Degradation(), rng) is None
    raw = u3.features_for(counts_only, ChainConfig("raw", 1e-6), u3.Degradation(), rng)
    assert raw is not None and raw.shape == (4000, 12)


def test_figures_are_written(summary: dict, tmp_path: Path) -> None:
    for plot in (plot_pareto, plot_power, plot_bridge):
        path = plot(summary, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000
