"""U5 end to end on synthetic inputs: no downloaded data, no ngspice."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis import u2, u3, u4, u5
from neurochain.analysis.u5_plots import (
    plot_artifact,
    plot_cancellation,
    plot_decoding,
    plot_recovery,
)
from neurochain.data.mearec import synthetic_recording
from neurochain.frontend import spice
from neurochain.paths import repo_root
from neurochain.power import ChainConfig, PowerModel, chain_power

CONVENTIONAL = ChainConfig("crossings", 10e-6, 10, 30_000.0)
CHEAP = ChainConfig("sbp", 0.25e-6, 8, 2000.0)
SMALL = u5.U5Config(
    amplitudes_mv=(0.3, 30.0),
    scenarios_mm=(0.0, 20.0),
    train_s=(0.5, 1.0),
    pause_s=(0.25, 0.5),
    decoding=u4.U4Config(n_folds=3, n_boot=100, decoders=("wiener",)),
)
DETECTION = u2.U2Config(calibration_s=1.5, max_workers=1)


def decoding_set(seed: int = 0) -> u3.DecodingSet:
    """Twelve channels whose rates follow a smooth hidden velocity."""
    rng = np.random.default_rng(seed)
    n = 2400
    velocity = np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode="same")[:, None] * 3
    rate = np.exp(1.0 + 0.5 * velocity @ rng.standard_normal((1, 12)))
    return u3.DecodingSet(
        name="toy",
        crossings=rng.poisson(rate).astype(float),
        sbp=10.0 + 2.0 * rate / rate.mean() + 0.3 * rng.standard_normal(rate.shape),
        targets=velocity,
        groups=np.repeat(np.arange(n // 40), 40),
        bin_s=0.05,
        n_lags=3,
    )


def fake_u3_summary() -> dict:
    """The parts of ``results/u3_metrics.json`` that U5 reads."""
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
            "sbp_noise_ratio": 1.2,
        }
        for chain in (CONVENTIONAL, CHEAP)
    ]
    return {
        "config": {"seed": 0, "nomad_rebin": 3, "power": asdict(PowerModel())},
        "configurations": rows,
        "datasets": {"toy": {"baseline": 0, "winner": 1}},
    }


@pytest.fixture(scope="module")
def summary() -> dict:
    recording = synthetic_recording(n_channels=2, duration_s=8.0, seed=1)
    rng = np.random.default_rng(5)
    broadband = recording.signal_uv(0, 10.0)[: 6 * 30_000] + 200.0 * np.sin(
        2 * np.pi * 7.0 * np.arange(6 * 30_000) / 30_000.0 + rng.uniform(0, 6)
    )
    return u5.compute_u5(
        recording,
        config=SMALL,
        detection=DETECTION,
        u3_summary=fake_u3_summary(),
        data=decoding_set(),
        broadband=(broadband, 30_000.0),
        with_spice=False,
    )


def test_summary_is_json_and_complete(summary: dict) -> None:
    text = json.dumps(summary)
    assert "NaN" not in text
    assert set(summary) >= {
        "config", "propagation", "recovery", "signal", "fixed_point", "in_vivo", "decoding",
        "cost", "key",
    }  # fmt: skip


def test_propagation_summary(summary: dict) -> None:
    same, far = summary["propagation"]["scenarios"]
    assert same["name"] == "same array" and same["stim_channel"] is not None
    assert far["stim_channel"] is None
    assert same["median_mv"] > 5 * far["median_mv"]
    assert same["share_above_adc"] == 1.0 and far["share_above_adc"] == 0.0
    assert len(same["amplitudes_mv"]) == 96
    bipolar = np.array(summary["propagation"]["bipolar_mv"])
    monopolar = np.array(summary["propagation"]["monopolar_mv"])
    assert np.all(bipolar < monopolar)


def test_recovery_summary(summary: dict) -> None:
    chains = summary["recovery"]["chains"]
    assert chains["sbp"]["model_ms"][1] > chains["crossings"]["model_ms"][1]
    assert chains["sbp"]["spice_ms"] == [None, None]
    assert chains["sbp"]["tau_ms"] == pytest.approx(0.53, abs=0.01)
    assert chains["crossings"]["peak_mv"][1] == pytest.approx(
        chains["crossings"]["limit_mv"], rel=0.02
    )
    assert summary["key"]["recovery_ms"]["sbp"] == chains["sbp"]["model_ms"][1]


def test_treatments_do_what_they_claim(summary: dict) -> None:
    sbp = summary["signal"]["sbp"]
    large = 1
    assert sbp["none"]["bias"][large] > 5.0
    assert abs(sbp["blank"]["bias"][large]) < 0.3
    assert abs(sbp["template"]["bias"][large]) < 0.3
    assert sbp["none"]["valid"][large] == 1.0
    assert sbp["template"]["valid"][large] > sbp["blank"]["valid"][large] + 0.1
    assert sbp["template"]["residual_uv"][large] < 0.05 * sbp["none"]["residual_uv"][large]
    crossings = summary["signal"]["crossings"]
    assert crossings["none"]["extra_rate_hz"][large] > 20.0
    assert crossings["blank"]["extra_rate_hz"][large] < 5.0
    assert crossings["blank"]["p_keep"][large] < crossings["template"]["p_keep"][large] <= 1.0
    windows = summary["signal"]["blank_samples"]
    assert windows["sbp"][0] <= windows["sbp"][1]
    assert summary["signal"]["n_taps"] == {"sbp": 16, "crossings": 300}


def test_fixed_point_and_real_recording(summary: dict) -> None:
    fixed = summary["fixed_point"]
    assert fixed["masks_identical"]
    assert fixed["canceller_max_difference_lsb"] < 0.6
    assert fixed["sbp_correlation"] > 0.98
    real = summary["in_vivo"]["treatments"]
    assert real["none"]["bias"] > 5.0
    assert abs(real["template"]["bias"]) < 0.5
    trace = summary["in_vivo"]["trace"]
    assert len(trace["time_ms"]) == len(trace["raw_uv"]) == len(trace["template_valid"])
    assert max(trace["raw_uv"]) > 900.0


def test_decoding_rows(summary: dict) -> None:
    decode = summary["decoding"]
    rows = {(r["chain"], r["scenario"], r["treatment"], r["fitted"]): r for r in decode["rows"]}
    assert len(decode["rows"]) == 2 * (2 * 3 + 3)
    for chain in u5.CHAINS:
        none = rows[chain, "same array", "none", "without stimulation"]
        template = rows[chain, "same array", "template", "without stimulation"]
        assert none["r2"] < template["r2"]
        assert none["loss"] < -0.2
        assert template["r2_low"] <= template["r2"] <= template["r2_high"]
        assert template["loss"] == pytest.approx(
            template["r2"] - decode["no_stimulation_r2"][chain]["wiener"], abs=1e-9
        )
    assert ("sbp", "20 mm away", "none", "with stimulation") not in rows
    assert 0 < decode["stimulated_bins"] < decode["n_bins"]
    assert (
        summary["key"]["sbp_template"]["r2"]
        == rows["sbp", "same array", "template", "without stimulation"]["r2"]
    )


def test_stimulated_features_change_only_stimulated_bins() -> None:
    measured = {
        "amplitudes_mv": [1.0, 10.0],
        "sbp": {"blank": {"bias": [0.5, 0.5], "sd": [0.0, 0.0]}},
        "crossings": {"blank": {"p_keep": [0.0, 0.0], "extra_rate_hz": [0.0, 0.0]}},
    }
    rng = np.random.default_rng(0)
    features = 10.0 + rng.random((200, 4))
    on = np.zeros(200, dtype=bool)
    on[50:100] = True
    amplitude = np.full(4, 3000.0)
    sbp = u5.stimulated_features("sbp", features, on, amplitude, 2, "blank", measured, rng)
    assert np.array_equal(sbp[~on], features[~on])
    assert np.all(sbp[on][:, [0, 1, 3]] > features[on][:, [0, 1, 3]] + 4.0)
    assert np.all(sbp[on, 2] == np.median(features[:, 2]))
    counts = u5.stimulated_features(
        "crossings", np.rint(features), on, amplitude, None, "blank", measured, rng
    )
    assert not counts[on].any() and counts[~on].all()


def test_cost(summary: dict) -> None:
    cost = summary["cost"]["chains"]
    assert cost["sbp"]["canceller_uw"] == pytest.approx(0.004)
    assert cost["sbp"]["chain_uw"] == pytest.approx(0.484, abs=1e-3)
    assert cost["crossings"]["canceller_uw"] == pytest.approx(15 * cost["sbp"]["canceller_uw"])
    assert cost["sbp"]["template_bits_per_channel"] == 16 * 14


def test_scenario_amplitudes() -> None:
    amplitude, stim = u5.scenario_amplitudes_uv(0.0, u5.U5Config())
    assert stim is not None and amplitude.argmax() == stim
    nearest = np.sort(amplitude)[-2]
    assert nearest == pytest.approx(60.0 / (4 * np.pi * 0.3 * 400e-6), rel=1e-6)
    far, none = u5.scenario_amplitudes_uv(20.0, u5.U5Config())
    assert none is None and far.max() < 1000.0


def test_unknown_treatment_is_rejected() -> None:
    chain = u5.chains(u5.U5Config())["sbp"]
    with pytest.raises(ValueError, match="unknown treatment"):
        u5.treat(np.zeros((1, 8), dtype=np.int64), np.array([2]), "magic", chain, SMALL, 4, [2])


def test_figures_and_table(summary: dict, tmp_path: Path) -> None:
    import importlib.util

    for plot in (plot_artifact, plot_recovery, plot_cancellation, plot_decoding):
        path = plot(summary, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000
    spec = importlib.util.spec_from_file_location(
        "update_readme", repo_root() / "scripts" / "update_readme.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.u5_table(summary)
    assert "Recovery time, 0.48 µW chain" in table and "ngspice not run" in table
    assert "Canceller cost" in table


def test_committed_netlist_matches_the_generator() -> None:
    path = repo_root() / "spice" / "u5_saturation.cir"
    assert path.read_text() == u5.reference_netlist(), "spice/u5_saturation.cir is stale"


@pytest.mark.skipif(not spice.ngspice_available(), reason="ngspice not installed")
def test_recovery_with_ngspice_agrees() -> None:
    result = u5.recovery(u5.U5Config(amplitudes_mv=(3.0, 30.0)), with_spice=True)
    for chain in result["chains"].values():
        assert chain["largest_model_spice_gap_us"] < 20.0
        assert np.allclose(chain["spice_ms"], chain["model_ms"], atol=0.02)
        assert chain["example"]["spice_uv"] is not None
