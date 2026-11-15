"""U6 on the real inputs: the committed metrics against U3, U4, U5 and a recomputation."""

import json

import numpy as np
import pytest

from neurochain.analysis import u3, u6
from neurochain.data import mearec
from neurochain.paths import results_dir

HAVE_MEAREC = (mearec.mearec_dir() / mearec.DEFAULT_FILE).is_file()
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_MEAREC, reason="MEArec recording not generated"),
]


def _committed(name: str) -> dict:
    return json.loads((results_dir() / name).read_text())


def test_week_zero_is_u4_and_the_untreated_rows_are_u4_and_u5() -> None:
    m = _committed("u6_metrics.json")
    head = _committed("u4_metrics.json")["headline"]["nason"]["decoders"]
    curves = m["key"]["curves"]
    for decoder in ("wiener", "kalman", "gru"):
        assert curves["nason_sbp"][decoder]["week0"] == pytest.approx(
            head[decoder]["winner_r2"], abs=1e-6
        )
        assert curves["nason_crossings"][decoder]["week0"] == pytest.approx(
            head[decoder]["conventional_r2"], abs=1e-6
        )
    forward = {
        r["decoder"]: r["r2"]
        for r in _committed("u4_metrics.json")["cross_session"]["rows"]
        if r["direction"] == "forward"
    }
    for row in m["cross_session"]["rows"]:
        if row["mode"] == "fixed":
            assert row["r2"] == pytest.approx(forward[row["decoder"]], abs=1e-6)
    u5_rows = {
        (r["decoder"], r["fitted"]): r["r2"]
        for r in _committed("u5_metrics.json")["decoding"]["rows"]
        if (r["chain"], r["scenario"], r["treatment"]) == ("sbp", "same array", "template")
    }
    for row in m["stimulation"]["rows"]:
        if row["mode"] == "fixed":
            assert row["r2"] == pytest.approx(
                u5_rows[row["decoder"], "without stimulation"], abs=1e-6
            )
        if row["mode"] == "refit":
            assert row["r2"] == pytest.approx(u5_rows[row["decoder"], "with stimulation"], abs=1e-6)


def test_committed_model_and_response_are_reproduced() -> None:
    m = _committed("u6_metrics.json")
    config = u6.U6Config()
    recording = mearec.load_mearec()
    background = u3.background_psd(recording, config.noise_uv)
    # Week 0 of the SBP chain is U3's winner.
    u3_metrics = _committed("u3_metrics.json")
    winner = u3_metrics["configurations"][u3_metrics["datasets"]["nason"]["winner"]]
    fresh = u6.chain_state(config, u6.FRESH, background)
    assert fresh["sbp_noise_ratio"] == pytest.approx(winner["sbp_noise_ratio"], rel=1e-9)
    assert m["model"]["states"][0]["sbp_noise_ratio"] == pytest.approx(fresh["sbp_noise_ratio"])
    settled = u6.state_of(config.aging, config.key_week)
    response = u6.aging_response(recording, config, (u6.FRESH, settled), background)
    recomputed = response.summary(rethreshold=True)["states"]
    committed = {
        (s["impedance_factor"], s["background_factor"], s["encapsulation"]): s
        for s in m["response"]["states"]
    }
    for state in recomputed:
        key = (state["impedance_factor"], state["background_factor"], state["encapsulation"])
        assert np.allclose(state["p_keep"], committed[key]["p_keep"])
        assert np.allclose(state["extra_rate_hz"], committed[key]["extra_rate_hz"])
