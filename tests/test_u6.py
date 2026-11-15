"""U6 end to end on synthetic inputs: no downloaded data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis import u2, u3, u4, u5, u6
from neurochain.analysis.u6_plots import (
    plot_cross_session,
    plot_model,
    plot_robustness,
    plot_stimulation,
    plot_weeks,
)
from neurochain.bridge import degrade_sbp, sbp_floor
from neurochain.data.mearec import synthetic_recording
from neurochain.decode.evaluate import r2_score
from neurochain.paths import repo_root
from neurochain.power import NARROWBAND_HZ

SMALL = u6.U6Config(
    weeks=(0.0, 39.0, 104.0),
    amplitude_grid=(1.0, 0.5, 0.0),
    check_amplitudes=(1.0, 0.5),
    shift=6,
    n_latents=3,
    align_calibration_s=40.0,
    decoding=u4.U4Config(n_folds=3, n_boot=100, decoders=("wiener",)),
)
DETECTION = u2.U2Config(calibration_s=1.5, max_workers=1)


def decoding_set(seed: int = 0, name: str = "toy", turnover: int = 0) -> u3.DecodingSet:
    """Twelve channels whose rates follow a smooth hidden velocity.

    ``turnover`` channels get a new, unrelated tuning, as if they now recorded
    other neurons; the rest keep the tuning of ``turnover == 0``.
    """
    rng = np.random.default_rng(seed)
    n = 2400
    velocity = np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode="same")[:, None] * 3
    tuning = np.random.default_rng(99).standard_normal((1, 12))
    tuning[0, :turnover] = np.random.default_rng(seed + 7).standard_normal(turnover)
    rate = np.exp(1.0 + 0.5 * velocity @ tuning)
    return u3.DecodingSet(
        name=name,
        crossings=rng.poisson(rate).astype(float),
        sbp=10.0 + 2.0 * rate / rate.mean() + 0.3 * rng.standard_normal(rate.shape),
        targets=velocity,
        groups=np.repeat(np.arange(n // 40), 40),
        bin_s=0.05,
        n_lags=3,
    )


STIM_SIGNAL = {
    "amplitudes_mv": [0.3, 30.0],
    "sbp": {"template": {"bias": [1.0, 3.0], "sd": [0.1, 0.3]}},
}


@pytest.fixture(scope="module")
def recording():  # type: ignore[no-untyped-def]
    return synthetic_recording(n_channels=2, duration_s=8.0, seed=1)


@pytest.fixture(scope="module")
def summary(recording) -> dict:  # type: ignore[no-untyped-def]
    data = decoding_set()
    return u6.compute_u6(
        recording,
        config=SMALL,
        detection=DETECTION,
        sets={"toy": data},
        session_pair=(data, decoding_set(seed=3, name="toy_late", turnover=6)),
        days_apart=273.9,
        stim_data=data,
        u5_signal=STIM_SIGNAL,
        broadband=(recording.signal_uv(0, 13.0), recording.fs_hz),
    )


def test_summary_is_json_and_complete(summary: dict) -> None:
    text = json.dumps(summary)
    assert "NaN" not in text
    assert set(summary) >= {
        "config", "model", "response", "response_fixed_threshold", "bridge_check", "cost",
        "variations", "in_vivo", "decoding", "cross_session", "stimulation", "key",
    }  # fmt: skip
    assert summary["variations"] == list(u6.variations(SMALL))


def test_model_follows_its_parameters(summary: dict) -> None:
    first, _, last = summary["model"]["states"]
    assert first["sbp_gain"] == 1.0 and first["crossings_gain"] == 1.0
    assert last["z_1khz_kohm"] == pytest.approx(2.48 * first["z_1khz_kohm"], rel=1e-3)
    assert last["rs_kohm"] > 10 * first["rs_kohm"]
    # A resistive sheath adds thermal noise and costs a little gain.
    assert last["electrode_noise_uv"] > 2 * first["electrode_noise_uv"]
    assert last["amplifier_noise_uv"] == pytest.approx(first["amplifier_noise_uv"], rel=0.02)
    assert last["frontend_noise_uv"] > first["frontend_noise_uv"]
    assert 0.9 < last["sbp_gain"] < 1.0 and 0.9 < last["crossings_gain"] < 1.0
    assert last["sbp_noise_ratio"] > first["sbp_noise_ratio"] >= 1.0
    amplitude = summary["model"]["amplitude"]
    assert amplitude["p50"][0] == 1.0
    assert amplitude["p10"][-1] < amplitude["p50"][-1] < amplitude["p90"][-1] < 1.0


def test_states_and_variations() -> None:
    model = SMALL.aging
    assert u6.state_of(model, 0.0) == u6.FRESH
    assert u6.state_of(model, 39.0) == u6.state_of(model, 104.0) == (2.48, 1.0, "series")
    varied = u6.variations(SMALL)
    assert u6.state_of(varied["no impedance rise"].aging, 39.0) == u6.FRESH
    assert u6.state_of(varied["rise in the interface, not a sheath"].aging, 39.0)[2] == "interface"
    assert varied["thresholds fixed at week 0"].rethreshold is False
    assert varied["amplitude decline doubled"].aging.amplitude_decline_per_month == 0.048


def test_smaller_spikes_lose_crossings(summary: dict) -> None:
    states = summary["response"]["states"]
    fresh = next(s for s in states if s["impedance_factor"] == 1.0)
    assert fresh["p_keep"][0] == 1.0 and fresh["extra_rate_hz"][0] == 0.0
    # The white background is about 19 µV over the whole band, so the threshold
    # sits near 85 µV: 100 µV spikes cross it, halved ones mostly do not.
    assert fresh["p_keep"][1] < 0.5
    assert fresh["p_keep"][2] < 0.1
    for state in states:
        assert state["p_keep"][0] >= state["p_keep"][1] >= state["p_keep"][2]
    check = summary["bridge_check"]["amplitudes"]["0.5"]
    for feature in ("sbp", "crossings"):
        assert check[f"{feature}_predicted"]["mean"] == pytest.approx(
            check[f"{feature}_actual"]["mean"], rel=0.15
        )


def test_week_zero_features_are_the_ones_u4_used(recording) -> None:  # type: ignore[no-untyped-def]
    data = decoding_set()
    background = u3.background_psd(recording, SMALL.noise_uv)
    response = u6.aging_response(
        recording, SMALL, (u6.FRESH, (2.48, 1.0, "series")), background, DETECTION
    )
    crossings = u6.make_ager("crossings", 12, SMALL.weeks, SMALL, response, background)
    assert np.array_equal(crossings.features(data, 0), data.crossings)
    assert np.all(crossings.p_keep[0] == 1.0) and np.all(crossings.extra_rate_hz[0] == 0.0)
    assert crossings.features(data, 2).mean() < data.crossings.mean()
    sbp = u6.make_ager("sbp", 12, SMALL.weeks, SMALL, response, background)
    n_effective = 2.0 * (NARROWBAND_HZ[1] - NARROWBAND_HZ[0]) * data.bin_s
    assert data.sbp is not None
    expected = degrade_sbp(
        data.sbp,
        float(sbp.sbp_ratio[0]),
        sbp_floor(data.sbp, n_effective),
        n_effective=n_effective,
        rng=np.random.default_rng(sbp.seed),
    )
    assert np.allclose(sbp.features(data, 0), expected)
    assert sbp.gain[0] == 1.0 and sbp.gain[2] < 1.0
    assert np.all(np.diff(sbp.amplitude, axis=0) <= 0.0)
    with pytest.raises(ValueError, match="no recorded SBP"):
        sbp.features(
            u3.DecodingSet("x", data.crossings, None, data.targets, data.groups, 0.05, 3), 0
        )


def test_decoding_rows(summary: dict) -> None:
    rows = summary["decoding"]["rows"]
    assert set(summary["decoding"]["conditions"]) == {"toy_sbp", "toy_crossings"}
    baseline = [r for r in rows if r["variation"] == "baseline"]
    assert {r["mode"] for r in baseline} == {"fixed", "mean", "zscore", "refit"}
    for row in baseline:
        if row["week"] == 0.0 and row["mode"] in ("fixed", "refit"):
            assert row["loss"] == 0.0
        if row["mode"] == "fixed":
            assert row["gain"] == 0.0
        assert row["r2_low"] <= row["r2"] <= row["r2_high"]
    varied = {r["variation"] for r in rows} - {"baseline"}
    assert varied == set(summary["variations"])
    assert {r["week"] for r in rows if r["variation"] != "baseline"} == {39.0, 104.0}
    key = summary["key"]["curves"]["toy_crossings"]["wiener"]
    week0 = next(
        r for r in baseline
        if (r["condition"], r["week"], r["mode"]) == ("toy_crossings", 0.0, "fixed")
    )  # fmt: skip
    assert key["week0"] == week0["r2"] > 0.5
    assert key["last_fixed"]["r2"] < key["week0"]


def test_running_zscore_rescues_a_decoder_from_lost_modulation() -> None:
    data = decoding_set()
    cfg = SMALL.decoding
    folds = u6.fit_folds("wiener", data, data.crossings, cfg)
    clean = r2_score(data.targets, u6.predict_folds(folds, data, data.crossings, SMALL))
    # Every channel keeps a third of its modulation and sits on a new level.
    changed = 0.3 * data.crossings + 2.0
    fixed = r2_score(data.targets, u6.predict_folds(folds, data, changed, SMALL))
    centred = r2_score(data.targets, u6.predict_folds(folds, data, changed, SMALL, "mean"))
    zscore = r2_score(data.targets, u6.predict_folds(folds, data, changed, SMALL, "zscore"))
    warm = r2_score(
        data.targets,
        u6.predict_folds(folds, data, changed, u6.U6Config(shift=6, warm_start=True), "zscore"),
    )
    assert clean > 0.8
    assert fixed < clean - 0.3  # a linear decoder's output shrinks with its input
    assert zscore > fixed + 0.2  # the running scale puts the modulation back
    assert zscore > centred  # a running mean alone cannot
    assert warm >= zscore - 0.01


def test_cross_session_rows(summary: dict) -> None:
    cross = summary["cross_session"]
    rows = {r["mode"]: r for r in cross["rows"]}
    assert tuple(rows) == u6.CROSS_MODES
    assert rows["fixed"]["gain"] == 0.0
    assert rows["within"]["r2"] > rows["fixed"]["r2"]
    # Half the channels changed their tuning: re-finding the latent axes helps.
    assert rows["aligned"]["r2"] > rows["latent"]["r2"]
    assert cross["weeks_apart"] == pytest.approx(273.9 / 7.0)
    assert cross["calibration_s"] == pytest.approx(40.0)
    assert summary["key"]["cross_session"]["wiener"]["aligned"] == rows["aligned"]["r2"]
    shift = cross["shift"]
    assert shift["n_channels"] == 12 and 0.0 <= shift["share_rate_up"] <= 1.0


def test_stimulation_rows(summary: dict, recording) -> None:  # type: ignore[no-untyped-def]
    stim = summary["stimulation"]
    assert stim["scenario"] == "same array" and 0 < stim["stimulated_bins"] < stim["n_bins"]
    modes = {r["mode"] for r in stim["rows"]}
    assert modes == {"fixed", "zscore", "gated mean", "gated zscore", "refit", "no stimulation"}
    assert {r["decoder"] for r in stim["rows"]} == set(u5.U5Config().decoding.decoders)
    for decoder in u5.U5Config().decoding.decoders:
        rows = {r["mode"]: r for r in stim["rows"] if r["decoder"] == decoder}
        assert rows["fixed"]["gain"] == 0.0
        assert rows["fixed"]["loss"] < 0.0  # a shift the decoder never saw costs something
        assert rows["gated zscore"]["r2"] > rows["fixed"]["r2"]


def test_cost_is_small_and_in_the_power_model_units(summary: dict) -> None:
    cost = summary["cost"]
    assert cost["modes"]["zscore"]["operations_per_bin"] == 17
    assert cost["modes"]["zscore"]["uw_per_channel"] == pytest.approx(17 / 0.05 * 1e-12 * 1e6)
    assert cost["modes"]["mean"]["uw_per_channel"] < cost["modes"]["zscore"]["uw_per_channel"]
    assert cost["modes"]["aligned"]["operations_per_bin"] == SMALL.n_latents
    assert cost["chains"]["sbp"]["chain_uw"] == pytest.approx(0.484)
    assert cost["chains"]["sbp"]["zscore_share_of_chain"] < 0.01


def test_in_vivo_check(summary: dict) -> None:
    states = summary["in_vivo"]["states"]
    assert states["settled"]["spike_band_sigma_uv"] > states["fresh"]["spike_band_sigma_uv"]
    assert states["fresh"]["crossing_rate_hz"] > 0.0


def test_figures_and_table(summary: dict, tmp_path: Path) -> None:
    for plot in (plot_model, plot_weeks, plot_cross_session, plot_robustness, plot_stimulation):
        path = plot(summary, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000
    spec = importlib.util.spec_from_file_location(
        "update_readme", repo_root() / "scripts" / "update_readme.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.u6_table(summary)
    for expected in (
        "Electrode after encapsulation",
        "R², 0.48 µW chain",
        "Real session pair, Wiener",
        "Stimulation shift, Wiener",
        "Changed assumptions",
        "Stabiliser cost",
    ):
        assert expected in table


def test_response_needs_the_settled_state(recording) -> None:  # type: ignore[no-untyped-def]
    background = u3.background_psd(recording, SMALL.noise_uv)
    with pytest.raises(ValueError, match="settled state"):
        u6.aging_response(recording, SMALL, (u6.FRESH,), background, DETECTION)
